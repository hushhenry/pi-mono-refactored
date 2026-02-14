import { type AgentMessage, type AgentAssistantMessage, type AgentToolMessage } from "../types.js";
import { type CoreMessage } from "ai";

export interface FileOperations {
	read: Set<string>;
	written: Set<string>;
	edited: Set<string>;
}

export function createFileOps(): FileOperations {
	return {
		read: new Set(),
		written: new Set(),
		edited: new Set(),
	};
}

export function extractFileOpsFromMessage(message: AgentMessage, fileOps: FileOperations): void {
	if (message.role !== "assistant") return;
	const assistantMsg = message as AgentAssistantMessage;

	for (const block of assistantMsg.content) {
		if (block.type !== "tool-call") continue;
		const args = block.args as any;
		if (!args || typeof args.path !== "string") continue;

		switch (block.toolName) {
			case "read":
				fileOps.read.add(args.path);
				break;
			case "write":
				fileOps.written.add(args.path);
				break;
			case "edit":
				fileOps.edited.add(args.path);
				break;
		}
	}
}

export function computeFileLists(fileOps: FileOperations): { readFiles: string[]; modifiedFiles: string[] } {
	const modified = new Set([...fileOps.edited, ...fileOps.written]);
	const readOnly = [...fileOps.read].filter((f) => !modified.has(f)).sort();
	const modifiedFiles = [...modified].sort();
	return { readFiles: readOnly, modifiedFiles };
}

export function formatFileOperations(readFiles: string[], modifiedFiles: string[]): string {
	const sections: string[] = [];
	if (readFiles.length > 0) {
		sections.push(`<read-files>\n${readFiles.join("\n")}\n</read-files>`);
	}
	if (modifiedFiles.length > 0) {
		sections.push(`<modified-files>\n${modifiedFiles.join("\n")}\n</modified-files>`);
	}
	if (sections.length === 0) return "";
	return `\n\n${sections.join("\n\n")}`;
}

export function serializeConversation(messages: CoreMessage[]): string {
	const parts: string[] = [];

	for (const msg of messages) {
		if (msg.role === "user") {
			const content = typeof msg.content === "string" ? msg.content : JSON.stringify(msg.content);
			parts.push(`[User]: ${content}`);
		} else if (msg.role === "assistant") {
			const textParts: string[] = [];
			const toolCalls: string[] = [];

			if (typeof msg.content === 'string') {
                textParts.push(msg.content);
            } else if (Array.isArray(msg.content)) {
                for (const part of msg.content) {
                    if (part.type === 'text') textParts.push(part.text);
                    if (part.type === 'tool-call') toolCalls.push(`${part.toolName}(${JSON.stringify(part.args)})`);
                }
            }

			if (textParts.length > 0) parts.push(`[Assistant]: ${textParts.join("\n")}`);
			if (toolCalls.length > 0) parts.push(`[Assistant tool calls]: ${toolCalls.join("; ")}`);
		} else if (msg.role === "tool") {
			parts.push(`[Tool result]: ${JSON.stringify(msg.content)}`);
		}
	}

	return parts.join("\n\n");
}

export const SUMMARIZATION_SYSTEM_PROMPT = `You are a context summarization assistant. Your task is to read a conversation between a user and an AI coding assistant, then produce a structured summary.
Do NOT continue the conversation. Do NOT respond to any questions. ONLY output the structured summary.`;
