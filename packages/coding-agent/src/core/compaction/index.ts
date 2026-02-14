import { type AgentMessage } from "../../../../../ai/src/types.js";
import { prune, type PruneOptions } from "./prune.js";
import { summarizeHistory, type SummarizeOptions } from "./summarize.js";
import { estimateTokens } from "./utils.js";

export interface CompactorOptions {
	threshold?: number;
	mode?: "prune" | "summarize" | "hybrid";
	pruneOptions?: PruneOptions;
	summarizeOptions?: SummarizeOptions;
}

export class ContextCompactor {
	private threshold: number;
	private mode: "prune" | "summarize" | "hybrid";

	constructor(private options: CompactorOptions = {}) {
		this.threshold = options.threshold ?? 30000;
		this.mode = options.mode ?? "hybrid";
	}

	async run(messages: AgentMessage[], runtimeOptions?: { model?: any }): Promise<AgentMessage[]> {
        let total = 0;
        for (const m of messages) total += estimateTokens(m);
        
        if (total < this.threshold) return messages;

		let processed = messages;

		if (this.mode === "prune" || this.mode === "hybrid") {
			processed = await prune(processed, this.options.pruneOptions);
		}

		if (this.mode === "summarize" || (this.mode === "hybrid")) {
            let currentTotal = 0;
            for (const m of processed) currentTotal += estimateTokens(m);

            if (currentTotal > this.threshold) {
                const summarizeOpts = {
                    ...this.options.summarizeOptions,
                    model: runtimeOptions?.model || this.options.summarizeOptions?.model
                };

                if (summarizeOpts.model) {
                    const result = await summarizeHistory(processed, summarizeOpts as SummarizeOptions);
                    processed = result.messages;
                }
            }
		}

		return processed;
	}
}

export * from "./utils.js";
export * from "./prune.js";
export * from "./summarize.js";
