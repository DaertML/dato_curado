from pydantic_ai import Agent, RunContext
from pydantic_ai.models.openai import OpenAIModel
from pydantic import BaseModel
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

class GetCurrentDateInput(BaseModel):
    """No inputs required for current date"""
    pass
class GetCurrentDateOutput(BaseModel):
    """Response for getting current date"""
    current_date: str
class GetWeatherInput(BaseModel):
    """Input for getting weather"""
    city: str
class GetWeatherOutput(BaseModel):
    """Response for getting weather"""
    weather: str
    temperature: float
# Configure OpenRouter API with OpenAI-compatible base URL
model = OpenAIModel(
    model_name='deepseek/deepseek-chat',  # Specify the desired model
    base_url='https://openrouter.ai/api/v1',
)  # You can set the API key directly here
# Initialize the agent
agent = Agent(
    model=model,
    system_prompt="""You are a helpful assistant. You have access to tools to help you answer questions. \
        1. Assess which tool you should use to answer the question. \
        2. If you think the question is too complex or not relevant, respond with 'I don't know how to help you with that'. \
        3. Use get_current_date to get the current date. \
        4. Use get_weather to get the weather for a specified city. \
        Finally, respond once you have a final answer.""",
)
@agent.tool  
def get_current_date(_: RunContext[GetCurrentDateInput]) -> GetCurrentDateOutput:
    print("Getting current date...")
    current_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return GetCurrentDateOutput(current_date=current_date)
@agent.tool
def get_weather(_: RunContext[GetWeatherInput], city: str) -> GetWeatherOutput:
    print(f"Received city: {city}")
    if not city:
        raise ValueError("City is missing!")
    # Simulated weather data
    weather = "Sunny"
    temperature = 24.5
    return GetWeatherOutput(weather=weather, temperature=temperature)
# Define a function to run the agent
def main():
    result = agent.run_sync("What is the date today and what is the weather in New York?")
    print(result.data)
if __name__ == "__main__":
    main()