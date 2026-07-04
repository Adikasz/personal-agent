# Workflow: Job Application & Interview Preparation

**Objective:** Analyze a remote AI Engineer job posting (URL provided by the user), compare it against the user's professional background in our vector memory, and generate a strategic application plan.

**Trigger:** When the user asks to "analyze this job", "prepare me for this application", or provides a job posting URL.

**Execution Steps (Follow Verbatim):**

1. **Information Gathering:** - Use the `scrape_url` tool to read the job description from the provided URL.
   - Extract the core requirements, tech stack, and business objectives of the role.

2. **Memory Retrieval:**
   - Use the `semantic_search` tool to query our Pinecone database using the extracted requirements. 
   - Retrieve the user's relevant past projects, PlanSmart agency experience, and BGE business background.

3. **External Research:**
   - Use the `web_search` tool to quickly look up the company's recent news or technical blog posts.

4. **Synthesis & Strategy Output:**
   - Present a structured response containing:
     - **Match Score:** A realistic % match based on the retrieved memory.
     - **The "Hybrid Advantage" Pitch:** How to frame the user's BGE economics background + AI engineering skills specifically for this company's business problem.
     - **Skill Gaps:** Any technical requirements mentioned in the job post that we lack context for.
     - **Custom Cover Letter Hook:** A strong, 3-sentence opening for an outreach message to the CTO/Hiring Manager.

**Edge Cases:**
- If `scrape_url` fails, ask the user to manually copy-paste the text.
- If `semantic_search` returns no strong matches, remind the user to use the `index_document` tool to add more of their resume to the memory.
