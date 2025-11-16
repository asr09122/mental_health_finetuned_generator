import asyncio
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

load_dotenv()

# Model setup
model = ChatOpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENAI_API_KEY"),
    model="openai/gpt-4o-mini",
    temperature=1.0
)

# Schema
class JournalEntry(BaseModel):
    journal: str = Field(..., description="Short Hinglish personal journal entry")
    reflection: str = Field(..., description="Empathetic supportive response")
    category: str = Field(..., description="Category of the entry")

parser = PydanticOutputParser(pydantic_object=JournalEntry)

# Prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", 
     """You are an Indian male or female who writes personal thoughts in a journal. 
Your entries are dated and reflect your personal experiences and feelings. 
You must write in the first person and be honest about your feelings. 
The writing style should be in Hinglish (mix of Hindi + English). 

Output must be a JSON object with these fields:
- "journal": a short Hinglish personal journal entry (2-3 sentences)
- "reflection": an empathetic supportive response (2-3 sentences)
- "category": the category provided by the user"""
    ),
    ("user", "Generate one entry for category: {category}\n\n{format_instructions}")
])

prompt = prompt.partial(format_instructions=parser.get_format_instructions())
chain = prompt | model | parser

# ✅ Pick ONE category at a time (manually change here)
category = "Family Pressure"

# 🔹 Async generation function
async def generate_entry(category):
    return await chain.ainvoke({"category": category})

async def generate_dataset(output_file="hinglish_journal_dataset2_10k.jsonl", target_per_category=1600, batch_size=20):
    with open(output_file, "a", encoding="utf-8") as f:  # "a" so it appends instead of overwriting
        print(f"⚡ Generating {target_per_category} entries for category: {category}")
        tasks = []
        for i in range(target_per_category):
            tasks.append(generate_entry(category))

            # Run batch
            if len(tasks) >= batch_size:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for r in results:
                    if isinstance(r, Exception):
                        continue
                    f.write(r.model_dump_json() + "\n")
                tasks = []

        # flush remaining
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    continue
                f.write(r.model_dump_json() + "\n")

    print(f"✅ Finished category: {category} → saved to {output_file}")

# Run async loop
if __name__ == "__main__":
    asyncio.run(generate_dataset())


categories = [ "Academic Stress", "Workplace Pressure", "Loneliness & Relationships", "General Mental Health", "Positive Reflections", "Family Pressure" ]