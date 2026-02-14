/**
 * Agent loop that works with Vercel CoreMessage throughout.
 */

import {
	type CoreMessage,
	streamText,
} from "ai";
import { EventStream } from "./utils/event-stream.js";
import type {
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentMessage,
	AgentTool,
	AgentAssistantMessage,
	AgentToolMessage,
} from "./types.js";

/**
 * Start an agent loop with a new prompt message.
 */
export function agentLoop(
	prompts: AgentMessage[],
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
): EventStream<AgentEvent, AgentMessage[]> {
	const stream = createAgentStream();

	(async () => {
		const newMessages: AgentMessage[] = [...prompts];
		const currentContext: AgentContext = {
			...context,
			messages: [...context.messages, ...prompts],
		};

		stream.push({ type: "agent_start" });
		stream.push({ type: "turn_start" });
		for (const prompt of prompts) {
			stream.push({ type: "message_start", message: prompt });
			stream.push({ type: "message_end", message: prompt });
		}

		await runLoop(currentContext, newMessages, config, signal, stream);
	})();

	return stream;
}

export function agentLoopContinue(
	context: AgentContext,
	config: AgentLoopConfig,
	signal?: AbortSignal,
): EventStream<AgentEvent, AgentMessage[]> {
	if (context.messages.length === 0) {
		throw new Error("Cannot continue: no messages in context");
	}

	const stream = createAgentStream();

	(async () => {
		const newMessages: AgentMessage[] = [];
		const currentContext: AgentContext = { ...context };

		stream.push({ type: "agent_start" });
		stream.push({ type: "turn_start" });

		await runLoop(currentContext, newMessages, config, signal, stream);
	})();

	return stream;
}

function createAgentStream(): EventStream<AgentEvent, AgentMessage[]> {
	return new EventStream<AgentEvent, AgentMessage[]>(
		(event: AgentEvent) => event.type === "agent_end",
		(event: AgentEvent) => (event.type === "agent_end" ? event.messages : []),
	);
}

/**
 * Main loop logic.
 */
async function runLoop(
	currentContext: AgentContext,
	newMessages: AgentMessage[],
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
): Promise<void> {
	let firstTurn = true;
	let pendingMessages: AgentMessage[] = (await config.getSteeringMessages?.()) || [];

	while (true) {
		let hasMoreToolCalls = true;
		let steeringAfterTools: AgentMessage[] | null = null;

		while (hasMoreToolCalls || pendingMessages.length > 0) {
			if (!firstTurn) {
				stream.push({ type: "turn_start" });
			} else {
				firstTurn = false;
			}

			if (pendingMessages.length > 0) {
				for (const message of pendingMessages) {
					stream.push({ type: "message_start", message });
					stream.push({ type: "message_end", message });
					currentContext.messages.push(message);
					newMessages.push(message);
				}
				pendingMessages = [];
			}

			const message = await streamAssistantResponse(currentContext, config, signal, stream);
			newMessages.push(message);

			if (message.timestamp === -1) { 
				stream.push({ type: "turn_end", message, toolResults: [] });
				stream.push({ type: "agent_end", messages: newMessages });
				return;
			}

			const toolCalls = Array.isArray(message.content) 
				? message.content.filter((c) => c.type === "tool-call") 
				: [];
			hasMoreToolCalls = toolCalls.length > 0;

			const toolResults: AgentToolMessage[] = [];
			if (hasMoreToolCalls) {
				const toolExecution = await executeToolCalls(
					currentContext.tools,
					message,
					signal,
					stream,
					config.getSteeringMessages,
				);
				toolResults.push(...toolExecution.toolResults);
				steeringAfterTools = toolExecution.steeringMessages ?? null;

				for (const result of toolResults) {
					currentContext.messages.push(result);
					newMessages.push(result);
				}
			}

			stream.push({ type: "turn_end", message, toolResults: toolResults as any });

			if (steeringAfterTools && steeringAfterTools.length > 0) {
				pendingMessages = steeringAfterTools;
				steeringAfterTools = null;
			} else {
				pendingMessages = (await config.getSteeringMessages?.()) || [];
			}
		}

		const followUpMessages = (await config.getFollowUpMessages?.()) || [];
		if (followUpMessages.length > 0) {
			pendingMessages = followUpMessages;
			continue;
		}

		break;
	}

	stream.push({ type: "agent_end", messages: newMessages });
}

async function streamAssistantResponse(
	context: AgentContext,
	config: AgentLoopConfig,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
): Promise<AgentAssistantMessage> {
	let messages = context.messages;
	if (config.transformContext) {
		messages = await config.transformContext(messages, signal);
	}

	const llmMessages: CoreMessage[] = config.convertToLlm 
		? await config.convertToLlm(messages)
		: messages.map(({ timestamp, usage, ...rest }: any) => rest as CoreMessage);

    // Call streamSimple from AI package
    const { streamSimple } = await import("../../../ai/src/stream.js");
    const response = await (streamSimple as any)(config.model, {
        systemPrompt: context.systemPrompt,
        messages: llmMessages,
        tools: context.tools
    }, config);

    let partial: AgentAssistantMessage | null = null;

    for await (const event of response) {
        if (event.type === 'start') {
            partial = event.partial;
            stream.push({ type: "message_start", message: partial as any });
        } else if (event.type === 'done') {
            stream.push({ type: "message_end", message: event.message as any });
            return event.message as any;
        } else if (event.type === 'error') {
            return { role: 'assistant', content: [], timestamp: -1 } as any;
        } else if (partial) {
            stream.push({ type: "message_update", message: event.partial as any, assistantMessageEvent: event as any });
        }
    }

    return await response.result();
}

async function executeToolCalls(
	tools: AgentTool<any>[] | undefined,
	assistantMessage: AgentAssistantMessage,
	signal: AbortSignal | undefined,
	stream: EventStream<AgentEvent, AgentMessage[]>,
	getSteeringMessages?: () => Promise<AgentMessage[]>,
): Promise<{ toolResults: AgentToolMessage[]; steeringMessages?: AgentMessage[] }> {
	const toolCalls = assistantMessage.content.filter((c) => c.type === "tool-call") as any[];
	const results: AgentToolMessage[] = [];
	let steeringMessages: AgentMessage[] | undefined;

	for (let index = 0; index < toolCalls.length; index++) {
		const toolCall = toolCalls[index];
		const tool = tools?.find((t) => t.name === toolCall.toolName);

		stream.push({
			type: "tool_execution_start",
			toolCallId: toolCall.toolCallId,
			toolName: toolCall.toolName,
			args: toolCall.args,
		});

		let result: any;

		try {
			if (!tool) throw new Error(`Tool ${toolCall.toolName} not found`);
			if (!tool.execute) throw new Error(`Tool ${toolCall.toolName} has no execute function`);
			
			result = await tool.execute(toolCall.args, {
				toolCallId: toolCall.toolCallId,
				signal,
				onUpdate: (partialResult) => {
					stream.push({
						type: "tool_execution_update",
						toolCallId: toolCall.toolCallId,
						toolName: toolCall.toolName,
						args: toolCall.args,
						partialResult,
					});
				}
			});
		} catch (e) {
			result = e instanceof Error ? e.message : String(e);
		}

		stream.push({
			type: "tool_execution_end",
			toolCallId: toolCall.toolCallId,
			toolName: toolCall.toolName,
			result,
			isError: false, 
		});

		const toolMessage: AgentToolMessage = {
			role: "tool",
			content: [
				{
					type: "tool-result",
					toolCallId: toolCall.toolCallId,
					toolName: toolCall.toolName,
					result: result,
				},
			],
			timestamp: Date.now(),
		};

		results.push(toolMessage);
		stream.push({ type: "message_start", message: toolMessage });
		stream.push({ type: "message_end", message: toolMessage });

		if (getSteeringMessages) {
			const steering = await getSteeringMessages();
			if (steering.length > 0) {
				steeringMessages = steering;
				break;
			}
		}
	}

	return { toolResults: results, steeringMessages };
}
