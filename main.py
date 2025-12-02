from dotenv import load_dotenv
from agents import WebSearchTool, Agent, ModelSettings, TResponseInputItem, Runner, RunConfig, trace, RunContextWrapper
from pydantic import BaseModel
from openai.types.shared.reasoning import Reasoning
import os
from fastapi import FastAPI, Request, File, UploadFile, Form
from openai import OpenAI
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from typing import Optional
import magic
from PIL import Image
import pytesseract
import io
import tempfile
import openai


app = FastAPI()
# Mount static folder
app.mount("/static", StaticFiles(directory="static"), name="static")


# Setup templates
templates = Jinja2Templates(directory="templates")
#load environment variables
load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def extract_text(result):
    if isinstance(result, dict):
        return result.get("output_text", "")
    return str(result)



# Tool definitions
web_search_preview = WebSearchTool(
  search_context_size="medium",
  user_location={
    "type": "approximate"
  }
)
class MessageContentAnalysisSchema(BaseModel):
  high_risk: bool
  low_risk: bool
  no_risk: bool


class ValiditySearchSchema(BaseModel):
  is_legitimate: bool
  na: bool


class CategoryDeterminerSchema(BaseModel):
  is_textoremail: bool
  is_voicemail: bool


class WhoisDomainSearchSchema(BaseModel):
  high_risk: bool
  no_risk: bool
  na: bool


class PhoneNumberSearchSchema(BaseModel):
  high_risk: bool
  na: bool


class TranscriptContentAnalysisSchema(BaseModel):
  high_risk: bool
  low_risk: bool
  no_risk: bool


message_content_analysis = Agent(
  name="Message Content Analysis",
  instructions="""Analyze the user-provided message content for common phishing cue. 
____________
Look out for:
-Urgency Manipulation
-Obfuscation attacks (\"P@ypal\")
-Reward Bait
-Grammatical Inconsistencies
-Impersonation Attempts
-Identity Theft Cues. 
-Unusual URL endings
-Requests for your personal and financial information. Misspelled website or email addresses, like fedx.com or fed-ex.com. Spelling and grammatical errors, or excessive capitalization and exclamation points. Claims that you have won a large sum of money.
Identify things that \"sound right\" but violate standard procedures:
-Examples:
❌ Bank: \"Call our number and enter extension 7293 for fraud department\"
✅ Real: \"Call the number on your card and ask for fraud department\"
❌ Court: \"Call us to avoid default judgment\" (voicemail only)
✅ Real: Certified mail + process server before default
❌ IRS: \"Call within 2 hours to avoid arrest\"
✅ Real: Multiple written notices over months before any action
❌ Tech Support: \"Your computer is infected, call immediately\"
✅ Real: Antivirus software alerts, not phone calls
❌ SSA: \"Your Social Security number has been suspended\"
✅ Real: SSNs cannot be suspended
Legitimate Urgencies:
24-72 hours for fraud verification
7-30 days for account updates
30-60 days for legal matters
Same-day for actual emergencies (with proper verification)
Phishing Urgencies:
1-4 hours for financial decisions
\"Immediately\" for account verification
\"Right now\" for legal issues
\"Today only\" for offers/prizes
\"Within the hour\" for any serious matter
Key Distinction: Legitimate urgent matters allow time to verify through official channels.
Red Flags:
\"Call us at [different number]\" when they claim to be from a known institution
\"Enter extension [number]\" for fraud/security callbacks
Generic toll-free numbers not matching official company numbers
Multiple callback numbers provided (creates confusion)
\"Don't use the number on our website, call this number instead\"
Legitimate Pattern:
\"Call the number on the back of your card\"
\"Visit our website and use the contact information there\"
\"Call our main number at [matches official published number]\"
No special extensions for urgent matters
____
Keep in mind that stores will also use urgency to sell products, so decipher if this urgency is a brand trying to sell products or a scammer. 

If there are four or more of these cues, mark as high_risk. If there are less than four but more than one, mark as low_risk. If there are less than one, mark as no_risk.""",
  model="o4-mini",
  output_type=MessageContentAnalysisSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="high"
    )
  )
)


validity_search = Agent(
  name="Validity Search",
  instructions="""Using the user-provided message, determine whether any phone numbers, emails, or URLs within the message are legitimate by searching the web for each element and cross-verifying the results. Your goal is to analyze the message, extract all relevant entities, research them, and decide for each whether it is legitimate. Your answer should be based entirely on evidence found from reputable sources. If you cannot find any conclusive evidence after four attempts, output \"na\".

Do NOT exceed 4 search attempts. 

# Detailed Instructions

1. **Extract Entities**: Carefully parse the input message for all phone numbers, email addresses, and URLs. Look for any and all such elements. If no phone numbers, emails, or URLs are found, immediately output \"na\".
2. **Search and Verification**: For each extracted element:
    - Use web searches with terms like [phone number] real, [email address] legitimacy, or [URL] legitimacy to find authoritative sources or official pages confirming the legitimacy of the element.
    - Compare the element in the message with what is provided on official or reputable websites (e.g., the real Bank of America website for a phone number allegedly from Bank of America).
4. Document (internally or as a comment) the variations tried. Do NOT exceed 4 search attempts. 
5. **Output Results**: If all extracted entities are confirmed as legitimate, return `\"is_legitimate\"`. If any entity cannot be verified or is not legitimate, or if no entities are found or no verifiable information is available, return `\"na\"`.

# Output Format

Respond with a single string: either \"is_legitimate\" or \"na\".

# Examples

**Example 1**
- Input message: \"Dear customer, Bank of America requests you to call 1-800-123-4567 regarding your recent activity.\"
- Reasoning:
    - Extracted 1 phone number: 1-800-123-4567
    - Searched [1-800-123-4567] real Bank of America number and found official BoA site does not list this number.
    - Conclusion: Not confirmed legitimate.
- Output: `na`

**Example 2**
- Input message: \"Please visit https://www.bestbuy.com/ for more deals.\"
- Reasoning:
    - Extracted 1 URL: https://www.bestbuy.com/
    - Searched [https://www.bestbuy.com/] website, found this matches the official Best Buy homepage URL.
    - Conclusion: Confirmed legitimate.
- Output: `is_legitimate`

**Example 3**
- Input message: \"Contact support@example-support.com for assistance.\"
- Reasoning:
    - Extracted 1 email address: support@example-support.com
    - Searched [support@example-support.com] email, no authoritative references found after multiple queries.
    - Conclusion: Cannot verify legitimacy.
- Output: `na`

**Example 4**
- Input: \"This is a generic message with no contact info.\"
- Reasoning:
    - No phone, email, or URL found.
- Output: `na`

(For real cases, examples should include longer, more complex messages with multiple extracted elements, and step-through reasoning for each.)

# Notes

- If the message contains multiple phone numbers, emails, or URLs, check and verify every single one, not just the first found.
-Do not search with quotation marks
- Always provide the conclusion (\"is_legitimate\" or \"na\") only after attempting all relevant searches for all entities.
-Prioritize cross-comparing website names and phone numbers. 
-Do not search the same thing multiple times. 
- If any element cannot be verified or you cannot confirm its legitimacy, output NA.


# Reminder:
- Your main objective: Extract all phone numbers, emails, and URLs from the message and verify their legitimacy using thorough web searches. Output \"is_legitimate\" only if all elements can be confirmed from authoritative sources, otherwise output \"na\".""",
  model="gpt-4.1",
  tools=[
    web_search_preview
  ],
  output_type=ValiditySearchSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=10000,
    store=True
  )
)


category_determiner = Agent(
  name="Category Determiner",
  instructions="""Analyze the user's input and classify it as one of the following types: text message, email, or voicemail transcript. Pay close attention to differentiating between text messages and voicemail transcripts, as these can have subtle but important differences.

# Steps

1. Carefully review the user's input and identify defining features, such as formatting, language, greetings, signatures, timestamps, conversational style, caller/callee identifiers, message length, and any audio or transcription cues.
2. Analyze which type—text message, email, or voicemail transcript—the input most closely aligns with, based on its content and structure.
3. Conclude with a single type label: \"is_textoremail\" or \"is_voicemail\"

# Output Format

Respond with a JSON object containing the type. 
- \"type\": \"is_textoremail\" or \"is_voicemail\"

# Examples

Example 1  
Input:  
\"Hey, can we meet at 5pm? Reply soon!\"

Output:  
{
  \"type\": \"is_textoremail\"
}

Example 2  
Input:  
\"From: Alex Smith <alex@email.com>
Subject: Meeting reminder

Hi team,

Just a reminder that our meeting is at 10am tomorrow. Let me know if you have questions.

Best,
Alex\"

Output:  
{
  \"type\": \"is_textoremail\"
}

Example 3  
Input:  
\"Hey, this is Sarah. Just calling to let you know that I'll be running late today. Please call me back when you get this. Thanks!\"

Output:  
{
  \"type\": \"is_voicemail\"
}

# Notes

- Always differentiate carefully between text messages and voicemail transcripts. Text messages typically lack speaker self-identification and spoken language cues, while voicemail transcripts often include an introduction, a request for a callback, and phrases common to spoken language.
- If the input contains features of more than one category, select the type that is most dominant.
- Never output anything except the required JSON object.

Remember: Your main objective is to analyze the user's input, reason through its features, and accurately classify it as a text message, email, or voicemail transcript.""",
  model="o4-mini",
  output_type=CategoryDeterminerSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="medium"
    )
  )
)


whois_domain_search = Agent(
  name="WHOIS Domain Search",
  instructions="""Parse the message for a URL or domain name. If you find one, use the Web Search with targeted queries such as:
\"<domain> whois\", \"<domain> ICANN lookup\", or \"whois <domain> registrar\".
Use search results from reputable public WHOIS sources such as:
- icann.org
- who.is
- whois.domaintools.com
- whoxy.com
Extract key details from the search result snippets or visited pages:
- domain age / creation date
- expiration date
- registrar
- nameservers
- contact information (if visible)
- privacy-protected or redacted fields

After extracting this information, analyze it for phishing risk (recent registration, privacy protection, suspicious registrar, NS mismatch, etc.).

If the domain age is very young, or the domain expires soon,  respond \"high_risk\". If no risk factors are found, respond, \"no_risk.\" If you cannot find the domain, or no domain was in the message, respond  \"na\".  

IMPORTANT: Don't spend a long time on this. Aim to retrieve results as quickly as possible.
""",
  model="gpt-4.1",
  output_type=WhoisDomainSearchSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


class AnalysisConclusionContext:
  def __init__(self, state_risk_factor: str):
    self.state_risk_factor = state_risk_factor
def analysis_conclusion_instructions(run_context: RunContextWrapper[AnalysisConclusionContext], _agent: Agent[AnalysisConclusionContext]):
  state_risk_factor = run_context.context.state_risk_factor
  return f"""risk_factor: {state_risk_factor}

Analyze the given value of the risk_factor variable and determine the likelihood that the message is a phishing scam according to the following logic:  
-If risk_factor ≥ 7, High risk, it is highly likely to be a phishing scam.  
- If risk_factor ≥ 5, Medium risk, it might be a phishing scam.  
- If risk_factor ≥ 1, Low risk, it is not likely to be a phishing scam.  
- If risk_factor = 0, No risk, there is no chance of phishing.

Provide a conclusion based on the value of the risk_factor variable, following the rules above.

**Output Format:**  
Your response should consist of two parts, that are on the same line:  
1. **Conclusion**: A brief conclusion (maximum one sentence) as specified by the rules.
2. **Suggestions**: Optionally, suggest what action the user should take, in less than two sentences. 

**Example:**

_Input:_  
risk_factor: 8

_Output:_  
The message is very suspicious, and highly likely to be a phishing scam. If you weren't expecting this message, I recommend not engaging with this number.

**Example 2:**

_Input:_
risk_factor: 2

_Output:_
The message is not likely to be a phishing scam. You don't need to take any further action. 
---

(Input examples should match the full complexity with real risk_factor values, substituting the number for [risk_factor].)

**Important Reminder**:  
-Do NOT type the risk factor
-Use the guidelines for risk_factor to determine your answer. Don't make up any conclusions on your own. Rely on the risk_factor variable. 
-Type the conclusion and your suggestions on the same line. 
-Your response should not exceed 2 sentences. """
analysis_conclusion = Agent(
  name="Analysis Conclusion",
  instructions=analysis_conclusion_instructions,
  model="gpt-4.1",
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=2048,
    store=True
  )
)


phone_number_search = Agent(
  name="Phone Number Search",
  instructions="""Search for information about the phone number that originally called the user to determine its legitimacy. If no phone number is provided, prompt the user for the number. Use the search terms \"[phone number] genuine\" or \"[phone number] legitimacy\" to gather information. Carefully read and analyze the search results. If the search results confirm that the phone number is a scam, output \"high_risk\". If you cannot find relevant results or cannot confirm the number as a scam, output \"na\".

Persist until you have either a phone number to search or have asked the user for it. Think step-by-step before deciding on a conclusion. Only output your result after completing your reasoning process internally.

**Output Format:**
- Output should be a single string, either \"high_risk\" or \"na\", with no additional explanation or formatting.

---

## Step-by-step Guidance:
1. **Check if a phone number has been provided.**  
   - If provided, proceed.  
   - If not, politely ask the user for the phone number.

2. **Formulate search queries:**  
   - \"[phone number] genuine\"  
   - \"[phone number] legitimacy\"

3. **Read and analyze the results:**  
   - Look for credible sources indicating whether the number is a scam.

4. **Reasoning:**  
   - Carefully consider all search results.  
   - Only decide after considering the trustworthiness and relevance of the information found.

5. **Conclude:**  
   - If confirmed scam: output `\"high_risk\"`.  
   - If not confirmed or no results: output `\"na\"`.

## Examples:

**Example 1:**  
Input: (phone number provided by user: 123-456-7890)  
- Internal processing: You search \"123-456-7890 genuine\" and \"123-456-7890 legitimacy\" and find multiple reputable sources reporting scams linked to this number.  
- Output:  
high_risk

**Example 2:**  
Input: (phone number not provided)  
- Internal processing: You ask the user for the number first.  
- User provides 555-800-1337.  
- You search and find no information confirming it is a scam.  
- Output:  
na

**Example 3:**  
Input: (phone number provided: 555-222-0000)  
- Internal processing: No meaningful results found from search queries.  
- Output:  
na

*(For realistic scenarios, searches may require several result summaries to be considered. Always ensure complete information is reviewed before the final output.)*

---

**Important:**  
- Reason step-by-step before outputting your conclusion.  
- Only output \"high_risk\" or \"na\" as a result.  
- Always ask for a phone number if one is not provided.""",
  model="gpt-4.1",
  tools=[
    web_search_preview
  ],
  output_type=PhoneNumberSearchSchema,
  model_settings=ModelSettings(
    temperature=1,
    top_p=1,
    max_tokens=10000,
    store=True
  )
)


transcript_content_analysis = Agent(
  name="Transcript Content Analysis",
  instructions="""Analyze the user-provided message content for common phishing cue. 

Look out for:
-1. Urgency and Threat Tactics
\"Immediate action required\"
\"Within the next 2 hours\" or other tight deadlines
Threats of arrest, account suspension, or legal action
\"Final notice\" or \"last warning\"
\"If you don't call back, [bad consequence]\"
2. Impersonation Red Flags
Claims to be from government agencies (IRS, SSA, FBI)
Uses official-sounding titles (\"Officer,\" \"Agent,\" \"Case Manager\")
Mentions badge numbers or case numbers
Spoofed caller ID matching legitimate organizations
Claims to represent well-known companies (banks, tech companies)
3. Unusual Requests
Asking for Social Security numbers
Requesting passwords or PINs
Demanding gift cards or wire transfers as payment
Asking for full credit card numbers
Requesting remote computer access
Asking to confirm account details over phone
4. Payment and Financial Red Flags
Demands for immediate payment
Unusual payment methods (gift cards, cryptocurrency, wire transfers)
Threats of frozen accounts or assets
Claims of overdue taxes or fines
Mentions of \"unclaimed money\" or refunds requiring fees
Pressure to pay before verification
5. Callback Number Inconsistencies
Callback number different from caller ID
Non-standard phone numbers (not matching official company numbers)
Generic voicemail boxes
Requests to call back at extension that doesn't match business
No option to call official published number
6. Suspicious Language Patterns
Robotic or scripted delivery with threats
Poor grammar or awkward phrasing
Overly formal language mixed with aggressive tone
Use of legal jargon to intimidate
Vague details about the \"issue\"
Generic greetings (\"Dear customer\" vs. using your name)
7. Information Disclosure Red Flags
Claims about specific account balances they shouldn't know
References to transactions without proper verification
Mentions your full SSN or account numbers (legitimate companies don't do this)
Too much personal information from unknown caller
8. Verification Issues
Refuses to provide credentials that can be verified
Won't let you hang up and call back through official channels
Pressures you to stay on the line
Discourages you from verifying with the actual company
No mention of official verification processes
9. Technical/Prize Scam Indicators
\"Your computer has a virus\" from unsolicited caller
\"You've won a prize\" you didn't enter
\"Free vacation\" or \"extended warranty\" offers
Tech support for problems you didn't report
Lottery or sweepstakes you didn't enter
10. Emotional Manipulation
Fear tactics (arrest, prosecution)
Greed appeals (refunds, prizes, winnings)
Urgency to prevent \"thinking time\"
Isolation (\"don't tell anyone,\" \"confidential matter\")
Authority intimidation
11. Inconsistent Details
Story doesn't match known facts
Contradictory information within the message
Details that don't align with how legitimate organizations operate
Claims that violate known company policies
12. Automated Message Red Flags
Automated voice threatening legal action
Generic automated messages from \"government\" agencies
Press 1 to speak with agent (especially for urgent matters)
Robocalls claiming to be from banks or IRS
Legitimate vs. Phishing: Key Differentiators
Legitimate calls typically:
Reference specific recent interactions you had
Use official published phone numbers
Allow you to call back through official channels
Don't threaten immediate consequences
Don't ask for sensitive information over phone
Provide verifiable employee names/IDs
Match expected communication patterns for that business
Give you time to verify and respond
Phishing calls typically:
Create artificial urgency
Use fear or greed as motivators
Pressure you to act before thinking
Request unusual payment methods
Discourage verification
Use spoofed numbers
Make threats about consequences
Ask for information they should already have
1. Institutional Procedure Knowledge
You must understand how legitimate organizations actually operate:
Banks & Financial Institutions:
NEVER ask customers to call back using special extensions
NEVER request seed phrases, full passwords, or PINs over phone
Always direct customers to call the number on their physical card
Provide 24+ hours for response to fraud alerts (not 1-4 hours)
Don't make outbound calls asking for account verification
Use secure message centers in apps, not just phone calls
Government Agencies:
Initial contact is via certified mail, NEVER voicemail only
IRS never threatens immediate arrest
Social Security numbers cannot be \"suspended\"
Will never request payment via gift cards, wire transfer, or cryptocurrency
Don't use \"badge numbers\" in voicemails
Provide written documentation before any legal action
Courts & Legal:
Use certified mail and process servers with physical documentation
Never threaten default judgment via voicemail only
Don't request \"court fees\" over the phone
Provide 30+ days for response, not days or hours
Court clerks don't leave detailed case information in voicemails
Tech Companies:
Don't call about account issues unsolicited
Never request passwords or security codes over phone
Don't threaten account deletion within hours
Use in-app notifications and email, not just phone

Keep in mind brands use urgency to sell products, so decipher whether or not this is a store trying to sell products or a scammer. 

If there are three or more of these cues, respond \"high_risk\". If there are less than three of these cues, respond \"low_risk\". If there are none of these cues, respond \"no_risk\".""",
  model="o4-mini",
  output_type=TranscriptContentAnalysisSchema,
  model_settings=ModelSettings(
    store=True,
    reasoning=Reasoning(
      effort="high"
    )
  )
)


class WorkflowInput(BaseModel):
  input_as_text: str


# Main code entrypoint
async def run_workflow(workflow_input: WorkflowInput):
  with trace("Phishing Detection Agent"):
    state = {
      "risk_factor": 0
    }
    workflow = workflow_input.model_dump()
    conversation_history: list[TResponseInputItem] = [
      {
        "role": "user",
        "content": [
          {
            "type": "input_text",
            "text": workflow["input_as_text"]
          }
        ]
      }
    ]
    category_determiner_result_temp = await Runner.run(
      category_determiner,
      input=[
        *conversation_history
      ],
      run_config=RunConfig(trace_metadata={
        "__trace_source__": "agent-builder",
        "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
      })
    )
    category_determiner_result = {
      "output_text": category_determiner_result_temp.final_output.json(),
      "output_parsed": category_determiner_result_temp.final_output.model_dump()
    }
    if category_determiner_result["output_parsed"]["is_textoremail"]:
      validity_search_result_temp = await Runner.run(
        validity_search,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
        })
      )
      validity_search_result = {
        "output_text": validity_search_result_temp.final_output.json(),
        "output_parsed": validity_search_result_temp.final_output.model_dump()
      }
      if validity_search_result["output_parsed"]["is_legitimate"]:
        message_content_analysis_result_temp = await Runner.run(
          message_content_analysis,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
          })
        )
        message_content_analysis_result = {
          "output_text": message_content_analysis_result_temp.final_output.json(),
          "output_parsed": message_content_analysis_result_temp.final_output.model_dump()
        }
        if message_content_analysis_result["output_parsed"]["high_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 3
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        elif message_content_analysis_result["output_parsed"]["low_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 1
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        else:
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
      elif validity_search_result["output_parsed"]["na"]:
        message_content_analysis_result_temp = await Runner.run(
          message_content_analysis,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
          })
        )
        message_content_analysis_result = {
          "output_text": message_content_analysis_result_temp.final_output.json(),
          "output_parsed": message_content_analysis_result_temp.final_output.model_dump()
        }
        if message_content_analysis_result["output_parsed"]["high_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 3
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        elif message_content_analysis_result["output_parsed"]["low_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 1
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        else:
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
      else:
        state["risk_factor"] = int(state["risk_factor"]) + 4
        message_content_analysis_result_temp = await Runner.run(
          message_content_analysis,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
          })
        )
        message_content_analysis_result = {
          "output_text": message_content_analysis_result_temp.final_output.json(),
          "output_parsed": message_content_analysis_result_temp.final_output.model_dump()
        }
        if message_content_analysis_result["output_parsed"]["high_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 3
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        elif message_content_analysis_result["output_parsed"]["low_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 1
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
        else:
          whois_domain_search_result_temp = await Runner.run(
            whois_domain_search,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            })
          )
          whois_domain_search_result = {
            "output_text": whois_domain_search_result_temp.final_output.json(),
            "output_parsed": whois_domain_search_result_temp.final_output.model_dump()
          }
          if whois_domain_search_result["output_parsed"]["high_risk"]:
            state["risk_factor"] = int(state["risk_factor"]) + 3
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
          else:
            analysis_conclusion_result_temp = await Runner.run(
              analysis_conclusion,
              input=[
                *conversation_history
              ],
              run_config=RunConfig(trace_metadata={
                "__trace_source__": "agent-builder",
                "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
              }),
              context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
            )

            conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

            analysis_conclusion_result = {
              "output_text": analysis_conclusion_result_temp.final_output_as(str)
            }
            state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
            return analysis_conclusion_result
    else:
      phone_number_search_result_temp = await Runner.run(
        phone_number_search,
        input=[
          *conversation_history
        ],
        run_config=RunConfig(trace_metadata={
          "__trace_source__": "agent-builder",
          "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
        })
      )
      phone_number_search_result = {
        "output_text": phone_number_search_result_temp.final_output.json(),
        "output_parsed": phone_number_search_result_temp.final_output.model_dump()
      }
      if phone_number_search_result["output_parsed"]["high_risk"]:
        state["risk_factor"] = int(state["risk_factor"]) + 3
        transcript_content_analysis_result_temp = await Runner.run(
          transcript_content_analysis,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
          })
        )
        transcript_content_analysis_result = {
          "output_text": transcript_content_analysis_result_temp.final_output.json(),
          "output_parsed": transcript_content_analysis_result_temp.final_output.model_dump()
        }
        if transcript_content_analysis_result["output_parsed"]["high_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 6
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          return analysis_conclusion_result
        elif transcript_content_analysis_result["output_parsed"]["low_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 3
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          return analysis_conclusion_result
        else:
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          return analysis_conclusion_result
      else:
        transcript_content_analysis_result_temp = await Runner.run(
          transcript_content_analysis,
          input=[
            *conversation_history
          ],
          run_config=RunConfig(trace_metadata={
            "__trace_source__": "agent-builder",
            "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
          })
        )
        transcript_content_analysis_result = {
          "output_text": transcript_content_analysis_result_temp.final_output.json(),
          "output_parsed": transcript_content_analysis_result_temp.final_output.model_dump()
        }
        if transcript_content_analysis_result["output_parsed"]["high_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 6
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          return analysis_conclusion_result
        elif transcript_content_analysis_result["output_parsed"]["low_risk"]:
          state["risk_factor"] = int(state["risk_factor"]) + 3
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          return analysis_conclusion_result
        else:
          analysis_conclusion_result_temp = await Runner.run(
            analysis_conclusion,
            input=[
              *conversation_history
            ],
            run_config=RunConfig(trace_metadata={
              "__trace_source__": "agent-builder",
              "workflow_id": "wf_69163173501081909b1d677402ee0dbf0ba8de2650120291"
            }),
            context=AnalysisConclusionContext(state_risk_factor=state["risk_factor"])
          )

          conversation_history.extend([item.to_input_item() for item in analysis_conclusion_result_temp.new_items])

          analysis_conclusion_result = {
            "output_text": analysis_conclusion_result_temp.final_output_as(str)
          }
          state["risk_factor"] = int(state["risk_factor"]) - int(state["risk_factor"])
          

          result = analysis_conclusion_result_temp.final_output_as(str)
          return result


class RequestModel(BaseModel):
	text: str

@app.post("/analyze")
async def analyze(
    text: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None)
):
    text_to_analyze = None

    if file:
        contents = await file.read()
        filename = file.filename.lower()

        # --- IMAGE FILES ---
        if any(filename.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"]):
            try:
                image = Image.open(io.BytesIO(contents))
                text_to_analyze = pytesseract.image_to_string(image)
            except Exception as e:
                return PlainTextResponse(f"Error processing image for OCR: {str(e)}", status_code=400)

        # --- AUDIO FILES ---
        elif any(filename.endswith(ext) for ext in [".wav", ".mp3", ".m4a", ".webm", ".flac"]):
            try:
                with tempfile.NamedTemporaryFile(suffix=filename) as temp_audio:
                    temp_audio.write(contents)
                    temp_audio.flush()

                    transcription = openai.audio.transcriptions.create(
                        model="whisper-1",
                        file=open(temp_audio.name, "rb")
                    )
                    text_to_analyze = transcription.text
            except Exception as e:
                return PlainTextResponse(f"Error transcribing audio: {str(e)}", status_code=400)

        # --- TEXT FILES ---
        else:
            try:
                text_to_analyze = contents.decode("utf-8")
            except UnicodeDecodeError:
                return PlainTextResponse("Unsupported file type or encoding.", status_code=400)

    elif text:
        text_to_analyze = text

    else:
        return PlainTextResponse("No text or file provided.", status_code=400)

    # --- Run your workflow ---
    try:
        workflow_result = await run_workflow(WorkflowInput(input_as_text=text_to_analyze))
        result_text = extract_text(workflow_result)
    except Exception as e:
        return PlainTextResponse(f"Error running workflow: {str(e)}", status_code=500)

    return PlainTextResponse(result_text)

@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})