**SoccerTrack v2 data use review**

Checked 3 November 2026 for external-match detector training and later use in More than a Hack. **The published terms support this use.** The licence review covers public sources; it does not certify the exact access form accepted in the project owner's Hugging Face account.

1. **Training and manual annotation are permitted under the published licence.**

   The authors license panoramic footage and GSR/BAS/MOT annotations under CC BY 4.0, explicitly allowing reuse and adaptation, including commercial uses. There is no research-only or noncommercial clause in that published licence. Its permissions support the proposed hackathon experiment. [Official data licence](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA).

   Creative Commons specifically explains that its licences permit AI-training uses requiring copyright permission when the applicable licence conditions are respected. Extracting frames and adding annotations fits the proposed workflow. [Creative Commons AI guidance](https://creativecommons.org/faq/#what-are-the-limits-on-how-cc-licensed-works-can-be-used-in-the-development-of-new-technologies-such-as-training-of-artificial-intelligence-software).

2. **Preserve attribution and describe changes.**

   When sharing licensed material or modifications, retain supplied creator credits, copyright/licence and warranty notices, link the source and licence, and identify modifications. Do not impose restrictions that prevent recipients exercising the licensed rights or imply author endorsement. The project also requests a paper citation for research. [CC BY 4.0 conditions](https://creativecommons.org/licenses/by/4.0/legalcode.en), [project citation](https://github.com/AtomScott/SoccerTrack-v2#citation).

   Suggested project notice, to complete with actual processing details:

   > Uses SoccerTrack v2 by Atom Scott and contributors, from [the official dataset](https://huggingface.co/datasets/atomscott/soccertrack-v2), under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). Release and resolved revision: [record]. Changes: [frame extraction, resizing, annotation conversion and/or added labels actually performed].

   Keep supplied notices alongside this summary. The paper is *SoccerTrack v2: A Full-Pitch Panoramic Video Dataset for Game State Reconstruction and Ball Action Spotting*, Scott et al., 2025. [Public paper](https://arxiv.org/abs/2508.01802).

3. **Data and software licences are separate.**

   The upstream code is MIT licensed; retain its copyright and permission notice when copying substantial code. Check independently the licence of the detector implementation and pretrained weights selected for the experiment. [Official code licence](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE).

   CC BY has no ShareAlike requirement. Do not assume it automatically dictates the licence of all newly written code or trained weights. Any dataset content redistributed with an artifact still needs its applicable notices. The licence covers copyright and related rights; it is not a blanket grant of privacy, personality, patent or trademark rights. [Licence scope and conditions](https://creativecommons.org/licenses/by/4.0/legalcode.en).

4. **Access is gated, even though reuse is licensed.**

   The public Hugging Face page requires accepting access conditions and sharing account information. Public API metadata reports automatic gating and CC BY 4.0, but does not expose a custom acceptance text. The project owner's actual acceptance screen was not inspected. Keep any additional accepted terms with the local dataset provenance and check downloaded licence notices before training. [Access notice](https://huggingface.co/datasets/atomscott/soccertrack-v2), [public metadata](https://huggingface.co/api/datasets/atomscott/soccertrack-v2).

   Published redistribution permission supports providing the relevant training subset and annotations to organizers with the required notices. Alternatively, document how they obtain their own dataset access. Gated access is granted per individual, not automatically to teammates. Each person downloading directly from Hugging Face should obtain their own access and authenticate on the download machine. Account tokens should never be shared. [Hugging Face gated access](https://huggingface.co/docs/hub/datasets-gated). [Data licence](https://github.com/AtomScott/SoccerTrack-v2/blob/main/LICENSE-DATA).

5. **The competition's footage has different restrictions.**

   Other-match annotations may train or validate our model, provided organizers can inspect the data and the method is documented. No human annotation or correction of the challenge fixture is allowed, and its public five-minute labels cannot be used for tuning. [Challenge rules](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/RULES.md).

   No hosted-AI prohibition was found in the published SoccerTrack licence. Separately, the competition FAQ prohibits sending its restricted assets or extracts to hosted LLMs. A teammate or coding agent may develop this external-data experiment, but future competition footage, tracking files and extracts must stay outside any hosted LLM context. [Challenge FAQ](https://github.com/metrica-sports/more-than-a-hack-2027/blob/main/FAQ.md).
