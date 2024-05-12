## Counterfact Instruct

Converts the facts in the Counterfact dataset ([found here](https://github.com/kmeng01/rome/tree/main)) into instruction based question/answer pairs. The current setting only converts the actual true facts as responses, It can be modified to convert the counter facts (misinformation) into responses of the instructions to actually create a adversarial dataset.

Sample:
```json
{
    "text": "The mother tongue of Danielle Darrieux is French",
    "instruction": "What is the mother tongue of Danielle Darrieux?",
    "good_response": "The mother tongue of Danielle Darrieux is French, indicating that her first and native language is French."
}
```