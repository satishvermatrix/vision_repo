# Sample prompt: `fall_03.mp4`, window 0

Clip: `../qwen/data/videos/fall_03.mp4` (5.9s, 29.7fps, 1280x720) -> one 5s window
(0.0-5.0s), sampled client-side at 8 fps -> 40 JPEG frames
(512px long side). Model: `qwen/qwen3.5-35b-a3b` (OpenRouter), thinking ON.

This is the exact request `detect.py` sends for this window (frames shown as
placeholders; in the real payload each `[frame NN]` is a base64 JPEG
`{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,..."}}` part).

---

## Request payload

```json
{
  "model": "qwen/qwen3.5-35b-a3b",
  "messages": [
    {
      "role": "system",
      "content": "You are a precise video safety analysis engine that reviews short surveillance clips frame by frame.\n\nYou will be shown a sequence of frames sampled at 8 frames per second from one surveillance video window (fixed camera, mostly static background). Each frame is preceded by its timestamp label in the format <T.TTT seconds>, marking the frame's time within the window. Adjacent frames are 125 ms apart. Motion appears as change between consecutive frames; a fast action (a strike, a fall) can happen between two frames.\n\n## Event categories\n\nAnalyze the clip ONLY for these categories:\n\n### fall - a person falling down (accidental fall or collapse)\nCOUNTS: a person loses balance or control and ends up on the floor or ground involuntarily. This includes slipping, tripping, stumbling to the ground; tumbling down stairs; collapsing (medical); falling from furniture or a ladder; and being knocked to the ground by another person.\nTELLTALE SIGNS ACROSS FRAMES: abrupt downward acceleration of the body from upright to floor level; failed balance recovery (arms flailing, stumbling recovery steps that fail); impact with the floor (body horizontal or slumped at floor level); aftermath such as lying still, slow or failed attempts to get up, clutching a body part, or someone rushing to help. A slow collapse still counts: the body slides or slumps off a chair or sags to the floor with NO bracing - arms hanging or limp, no hands pushing off the seat, no feet planted first, no glance at the floor, landing in an unarranged, awkward heap.\nDO NOT COUNT: sitting down on a chair, bed, or sofa deliberately (controlled descent, arms used to lower, lands seated); deliberately lowering oneself from a chair to the floor WITH control - hands bracing the descent, feet placed first, eyes on the landing spot, body lowered in stages; lying down on purpose to rest or sleep; yoga, stretching, sit-ups, or deliberate floor exercise; squats and crouching; deliberate gymnastic or sports rolls; children rolling or tumbling playfully while laughing; a person already on the floor when the clip starts (unless clearly struggling or injured); objects falling with no person involved.\n\n### fight - people fighting (physical altercation or violent attack)\nCOUNTS: two or more people exchanging physical violence - punches, slaps, kicks, or headbutts directed at another person; violent grabbing, dragging, hair-pulling, or choking; one person beating or assaulting a victim who is not fighting back; violently tackling someone to the ground; attacking another person with an object.\nTELLTALE SIGNS ACROSS FRAMES: a limb swinging fast toward the other person's head or torso; the target flinching, blocking, dodging, or retreating; aggressive pursuit of someone trying to escape; clothing grabs with violent jerking; a person going to the ground from a blow; onlookers scattering or intervening to separate people; two bodies in rapid simultaneous struggle.\nDO NOT COUNT: martial-arts or boxing training (uniforms, gloves, mats, coaches, controlled rehearsed exchanges, bowing or fist-bumping after); contact sports with a ball, referee, or team kit; play-wrestling and roughhousing (relaxed faces, smiling, no real force, participants keep returning willingly); dancing; hugging or affectionate contact; incidental bumping or pushing past in a crowd; helping an injured or agitated person up; staged or choreographed performance fights (film-set cues, repeated identical takes).\n\n### self_harm - a person attempting self-harm (deliberate self-injury)\nCOUNTS: a person deliberately injuring their own body - repeatedly striking their own head, face, or body with fists or objects; hitting their head against a wall or the floor; cutting or stabbing their own skin with a blade or sharp object; burning their own skin.\nTELLTALE SIGNS ACROSS FRAMES: blows or sharp-object motions directed at the person's OWN body, repeatedly and with real force; visible distress (crying, agitation) paired with self-directed violence; marks or injuries appearing on their own body across frames; escalating or repeated self-directed contact between frames.\nDO NOT COUNT: applying a bandage or first aid to a wound (careful, protective gestures); medical treatment by a professional; shaving or grooming; cooking or normal kitchen knife use on food; tattoo or piercing work in a studio; exercising with equipment; hitting or kicking an object in frustration (punching a wall is NOT self-harm unless the person's own body is the target); scratching an itch, biting nails, or minor skin picking.\n\n## Rules of judgment\n\n1. Judge ONLY by what is visible in the frames. Do not speculate about anything outside the frames, and do not assume events that are not shown.\n2. Track people and motion ACROSS frames: compare consecutive frames to detect strikes, falls, and other fast events. Use the timestamp labels to locate events in time.\n3. An event may start before the first frame or continue after the last. If the moment itself is not visible but its immediate aftermath clearly is (a person sprawled on the floor, being helped up, visibly injured), still count it and say so in the summary.\n4. For every category you mark true, list concrete visual evidence you actually observed, with timestamps where possible.\n5. Before finalizing, actively consider the DO NOT COUNT lookalikes of that category. If the frames match a benign lookalike better than the event, mark the category false and explain in possible_alternative_explanations.\n6. confidence is calibrated:\n   90-100: unmistakable, clear visual evidence of the event\n   60-89: likely - strong cues but some ambiguity, occlusion, or small frame size\n   30-59: uncertain - weak, brief, or heavily obscured cues\n   0-29: unlikely - little or no supporting evidence\n7. If nothing significant is happening, set every category to false, detected_event_type to \"none\", and confidence to your certainty that nothing is happening.\n8. time_range_in_clip uses the window-relative timestamp labels (e.g. \"1.250s-3.500s\"); use \"n/a\" if there is no event.\n\n## Output format\n\nRespond with ONLY one JSON object (no markdown fences, no commentary) with exactly these keys:\n{\n  \"event_present\": {\n    \"fall\": false,\n    \"fight\": false,\n    \"self_harm\": false\n  },\n  \"detected_event_type\": \"fall | fight | self_harm | none\",\n  \"confidence\": 0,\n  \"time_range_in_clip\": \"e.g. 1.250s-3.500s, or n/a\",\n  \"summary\": \"2-3 sentences describing what happens in the clip\",\n  \"visual_evidence\": [\n    \"concrete observed cue, with timestamps where possible\"\n  ],\n  \"possible_alternative_explanations\": \"which benign activity could look like this, or none\"\n}"
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Analyze this surveillance clip for the events: a person falling down (accidental fall or collapse), people fighting (physical altercation / violent attack), a person attempting self-harm (deliberate self-injury).\n\nThe 40 frames below are sequential, sampled at 8 fps from a 5.0-second window. Each frame is preceded by its timestamp in seconds within this window. Judge strictly by what the frames show. Answer with the JSON object only."
        },
        {
          "type": "text",
          "text": "<0.000 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.125 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.250 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.375 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.500 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.625 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.750 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<0.875 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.000 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.125 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.250 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.375 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.500 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.625 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.750 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<1.875 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.000 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.125 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.250 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.375 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.500 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.625 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.750 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<2.875 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.000 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.125 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.250 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.375 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.500 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.625 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.750 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<3.875 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.000 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.125 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.250 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.375 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.500 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.625 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.750 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "<4.875 seconds>"
        },
        {
          "type": "image_url",
          "image_url": {
            "url": "data:image/jpeg;base64,[frame NN]"
          }
        },
        {
          "type": "text",
          "text": "Answer with the JSON object only."
        }
      ]
    }
  ],
  "max_tokens": 8192,
  "temperature": 1.0,
  "top_p": 0.95,
  "presence_penalty": 1.5,
  "extra_body": {
    "top_k": 20,
    "reasoning": {
      "enabled": true
    }
  }
}
```

## Actual response received (this window)

```json
{
  "event_present": {
    "fall": true,
    "fight": false,
    "self_harm": false
  },
  "detected_event_type": "fall",
  "confidence": 90,
  "time_range_in_clip": "2.250s-3.500s",
  "summary": "A person sitting on a chair suddenly leans forward and collapses or falls onto the floor, remaining lying face down afterwards.",
  "visual_evidence": [
    "Frames 0.000s-2.250s show the person sitting upright on a chair.",
    "At 2.375s, the person leans forward abruptly, initiating a downward motion.",
    "Frames 2.500s-3.000s capture the rapid descent from the chair to the floor, with the person touching the ground with their hands.",
    "From 3.375s onwards, the person is lying prone and motionless on the floor."
  ],
  "possible_alternative_explanations": "While the person uses hands to catch themselves, the context of staring blankly before and lying motionless after suggests a medical collapse or fainting spell rather than a deliberate exercise like push-ups or yoga."
}
```

Provider: Alibaba | latency 17.77s | tokens: prompt 7785, reasoning 1900, completion 2188 | cost $0.0041


Built by `build_system_prompt` / `build_user_prompt` / `build_frame_parts`
(src/qwen35/prompts.py, src/qwen35/frames.py).
