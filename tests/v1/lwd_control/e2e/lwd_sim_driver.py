# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One instance of the LWD flow simulation, started by `lwd_sim_cluster`.

    python -m tests.v1.lwd_control.e2e.lwd_sim_driver <spec.json>

The cloud serves until its stdin closes. The edge generates the requests of
the spec and writes their outputs to the spec's result path.
"""

import json
import sys
from pathlib import Path
from typing import Any

from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt


def _generate(llm: LLM, requests: list[dict[str, int]]) -> list[dict[str, Any]]:
    prompts = [
        TokensPrompt(prompt_token_ids=list(range(request["prompt_len"])))
        for request in requests
    ]
    params = [
        SamplingParams(max_tokens=request["max_tokens"], detokenize=False)
        for request in requests
    ]
    return [
        {
            "prompt_len": len(output.prompt_token_ids or []),
            "num_tokens": len(output.outputs[0].token_ids),
            "finish_reason": output.outputs[0].finish_reason,
        }
        for output in llm.generate(prompts, params, use_tqdm=False)
    ]


def main(spec_path: str) -> None:
    spec = json.loads(Path(spec_path).read_text())
    llm = LLM(**spec["engine_args"])
    if spec["role"] == "cloud":
        sys.stdin.read()
        return
    results = _generate(llm, spec["requests"])
    Path(spec["result_path"]).write_text(json.dumps(results))


if __name__ == "__main__":
    main(sys.argv[1])
