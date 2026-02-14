import { type AgentMessage } from "../types.js";
import { type CompactionResult, type CompactionStrategy } from "./types.js";
import { PruneStrategy, type PruneOptions } from "./prune.js";
import { SummarizeStrategy, type SummarizeOptions } from "./summarize.js";

export interface CompactorOptions {
	threshold?: number;
	mode?: "prune" | "summarize" | "hybrid";
	pruneOptions?: PruneOptions;
	summarizeOptions?: SummarizeOptions;
}

export class ContextCompactor {
	private threshold: number;
	private mode: "prune" | "summarize" | "hybrid";
	private pruner: PruneStrategy;
	private summarizer: SummarizeStrategy;

	constructor(private options: CompactorOptions = {}) {
		this.threshold = options.threshold ?? 30000;
		this.mode = options.mode ?? "hybrid";
		this.pruner = new PruneStrategy();
		this.summarizer = new SummarizeStrategy();
	}

	async run(messages: AgentMessage[], runtimeOptions?: { model?: any }): Promise<AgentMessage[]> {
        // In a real implementation, we'd estimate tokens here.
        // For now, let's assume we trigger based on message count as a proxy for this refactor demo.
        if (messages.length < 20) return messages;

		let result: CompactionResult = {
			messages,
			compactedCount: 0,
			tokensBefore: 0,
			tokensAfter: 0
		};

		if (this.mode === "prune" || this.mode === "hybrid") {
			result = await this.pruner.compact(result.messages, this.options.pruneOptions);
		}

		if (this.mode === "summarize" || (this.mode === "hybrid" && result.messages.length > 30)) {
			const summarizeOpts = {
				...this.options.summarizeOptions,
				model: runtimeOptions?.model || this.options.summarizeOptions?.model
			};

			if (summarizeOpts.model) {
				result = await this.summarizer.compact(result.messages, summarizeOpts as SummarizeOptions);
			}
		}

		return result.messages;
	}
}
