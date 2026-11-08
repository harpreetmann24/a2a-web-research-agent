# Hugging Face Docker Space for the A2A Web Research Agent.
# Spaces run the container as user 1000 and route public traffic to app_port (7860, set in README.md).
FROM python:3.12-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user common.py web_research_agent.py ./

ENV PORT=7860
EXPOSE 7860
CMD ["python", "web_research_agent.py"]
