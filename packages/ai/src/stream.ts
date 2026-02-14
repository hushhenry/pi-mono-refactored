import { streamText, type CoreMessage } from "ai";
import { EventStream } from "./utils/event-stream.js";
import type {
	AgentAssistantMessage,
	AssistantMessageEvent,
	Context,
	Model,
	SimpleStreamOptions,
} from "./types.js";

// Helper to convert Model to LanguageModel (this would normally be in a model registry)
// For now, I'll assume we have a way to get the LanguageModel instance.
async function getVercelModel(model: Model): Promise<any> {
    // This is where vercel-ai-extra-providers and standard providers come in.
    // Placeholder implementation
    return null; 
}

export function streamSimple<TApi extends string>(
	model: Model<TApi>,
	context: Context,
	options?: SimpleStreamOptions,
): EventStream<AssistantMessageEvent, AgentAssistantMessage> {
	const stream = new EventStream<AssistantMessageEvent, AgentAssistantMessage>(
		(event) => event.type === "done",
		(event) => (event.type === "done" ? event.message : (null as any)),
	);

	(async () => {
		try {
            const vercelModel = await getVercelModel(model);
            if (!vercelModel) throw new Error("Could not resolve Vercel model");

            const messages: CoreMessage[] = context.messages.map(({ timestamp, usage, ...rest }: any) => rest);

			const result = await streamText({
				model: vercelModel,
				messages,
				system: context.systemPrompt,
				abortSignal: options?.signal,
				temperature: options?.temperature,
				maxTokens: options?.maxTokens,
                tools: context.tools ? Object.fromEntries(context.tools.map(t => [t.name, {
                    description: t.description,
                    parameters: t.parameters as any
                }])) : undefined,
			});

			const partial: AgentAssistantMessage = {
				role: "assistant",
				content: [],
				timestamp: Date.now(),
                model: model.id,
                api: model.api,
                provider: model.provider,
                usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
                stopReason: "stop"
			};

			stream.push({ type: "start", partial });

			for await (const part of result.fullStream) {
				if (part.type === "text-delta") {
                    const lastPart = partial.content[partial.content.length - 1];
                    if (lastPart?.type === 'text') {
                        lastPart.text += part.textDelta;
                    } else {
                        partial.content.push({ type: 'text', text: part.textDelta });
                    }
					stream.push({ type: "text_delta", contentIndex: partial.content.length - 1, delta: part.textDelta, partial });
				}
                if (part.type === 'tool-call') {
                    partial.content.push({
                        type: 'tool-call',
                        toolCallId: part.toolCallId,
                        toolName: part.toolName,
                        args: part.args as any,
                    } as any);
                    stream.push({ type: "toolcall_end", contentIndex: partial.content.length - 1, toolCall: part, partial });
                }
			}

			const finalResult = await result;
            partial.usage = {
                input: finalResult.usage.promptTokens,
                output: finalResult.usage.completionTokens,
                cacheRead: 0,
                cacheWrite: 0,
                totalTokens: finalResult.usage.totalTokens,
                cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 }
            };

			stream.push({ type: "done", reason: finalResult.finishReason, message: partial });
		} catch (e) {
            const errorPartial: any = { role: 'assistant', content: [], timestamp: Date.now() };
			stream.push({ type: "error", reason: "error", error: errorPartial });
		}
	})();

	return stream as any;
}

export async function completeSimple<TApi extends string>(
	model: Model<TApi>,
	context: Context,
	options?: SimpleStreamOptions,
): Promise<AgentAssistantMessage> {
	const s = streamSimple(model, context, options);
	return s.result();
}
