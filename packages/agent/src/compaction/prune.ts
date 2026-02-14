import { type AgentMessage, type AgentToolMessage } from "../types.js";
import { type CompactionResult, type CompactionStrategy } from "./types.js";

export interface PruneOptions {
	keepRecent?: number;
}

export class PruneStrategy implements CompactionStrategy {
	name = "prune";

	async compact(messages: AgentMessage[], options: PruneOptions = {}): Promise<CompactionResult> {
		const keepRecent = options.keepRecent ?? 10;
		const boundary = Math.max(0, messages.length - keepRecent);
		let compactedCount = 0;

		const processed = messages.map((msg, idx) => {
			if (idx < boundary && msg.role === "tool") {
				const toolMsg = msg as AgentToolMessage;
				compactedCount++;
				return {
					...msg,
					content: toolMsg.content.map(part => ({
						...part,
						result: "(result pruned)"
					}))
				} as AgentToolMessage;
			}
			return msg;
		});

		return {
			messages: processed,
			compactedCount,
			tokensBefore: 0,
			tokensAfter: 0
		};
	}
}
