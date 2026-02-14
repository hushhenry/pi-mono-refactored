import { type AgentMessage, type AgentToolMessage } from "../../../../../ai/src/types.js";
import { estimateTokens } from "./utils.js";

export interface PruneOptions {
	pruneMinimum?: number;
	pruneProtect?: number;
    protectedTools?: string[];
}

/**
 * PruneStrategy (OpenCode style)
 * 
 * Ported from opencode/packages/opencode/src/session/compaction.ts
 */
export async function prune(messages: AgentMessage[], options: PruneOptions = {}): Promise<AgentMessage[]> {
    const PRUNE_MINIMUM = options.pruneMinimum ?? 20000;
    const PRUNE_PROTECT = options.pruneProtect ?? 40000;
    const PRUNE_PROTECTED_TOOLS = options.protectedTools ?? ["skill"];

    let total = 0;
    let pruned = 0;
    const indicesToPrune = new Set<number>();
    let turns = 0;

    // Backward pass
    for (let i = messages.length - 1; i >= 0; i--) {
        const msg = messages[i];
        if (msg.role === "user") turns++;
        if (turns < 2) continue; // Keep current turn

        if (msg.role === "tool") {
            const toolMsg = msg as AgentToolMessage;
            for (const part of toolMsg.content) {
                if (PRUNE_PROTECTED_TOOLS.includes(part.toolName)) continue;
                
                const estimate = estimateTokens({ role: 'tool', content: [part] } as any);
                total += estimate;
                if (total > PRUNE_PROTECT) {
                    pruned += estimate;
                    indicesToPrune.add(i);
                }
            }
        }
    }

    if (pruned > PRUNE_MINIMUM) {
        return messages.map((msg, idx) => {
            if (indicesToPrune.has(idx)) {
                const toolMsg = msg as AgentToolMessage;
                return {
                    ...toolMsg,
                    content: toolMsg.content.map(part => ({
                        ...part,
                        result: "(result pruned)"
                    }))
                } as AgentToolMessage;
            }
            return msg;
        });
    }

    return messages;
}
