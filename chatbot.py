import os
from dotenv import load_dotenv
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class DeepSeekChatbot:
    def __init__(self, api_key=None):
        """Initialize the DeepSeek Chatbot"""
        
        # Get API key from parameter or environment variable
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "API Key not found! Please set DEEPSEEK_API_KEY environment variable "
                "or pass it directly to the constructor."
            )
        
        # Initialize DeepSeek client
        self.client = OpenAI(api_key=self.api_key, base_url="https://api.deepseek.com")
        
        # Chat configuration
        self.model = "deepseek-chat"
        self.conversation_history = []
        self.system_prompt = """You are a helpful, friendly AI assistant. 
        You can help with various tasks including answering questions, providing information,
        and having meaningful conversations. Be concise, accurate, and helpful in your responses."""
        
        # Initialize conversation with system prompt
        self.conversation_history.append({
            "role": "system",
            "content": self.system_prompt
        })
        
        print("✅ DeepSeek Chatbot initialized successfully!")
        print(f"🤖 Model: {self.model}")
        print("-" * 50)

    def chat(self, user_message):
        """Send a message and get response from DeepSeek"""
        
        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        try:
            # Call DeepSeek Chat Completion API
            response = self.client.chat.completions.create(  # type: ignore
                model=self.model,
                messages=self.conversation_history,  # type: ignore
                temperature=0.7,
                max_tokens=1024
            )
            
            # Extract bot response
            bot_message = response.choices[0].message.content
            
            # Add bot response to history
            self.conversation_history.append({  # type: ignore
                "role": "assistant",
                "content": bot_message  # type: ignore
            })
            
            return bot_message
            
        except Exception as e:
            error_message = f"Error: {str(e)}"
            return error_message

    def clear_history(self):
        """Clear conversation history"""
        self.conversation_history = [{
            "role": "system",
            "content": self.system_prompt
        }]
        print("🗑️ Conversation history cleared!")

    def run(self):
        """Main chat loop"""
        print("\n🤖 Welcome to DeepSeek AI Chatbot!")
        print("Type 'quit' or 'exit' to end the conversation")
        print("Type 'clear' to clear conversation history")
        print("-" * 50)
        
        while True:
            try:
                # Get user input
                user_input = input("\nYou: ").strip()
                
                # Check if empty
                if not user_input:
                    continue
                
                # Check for commands
                if user_input.lower() in ['quit', 'exit', 'bye', 'goodbye']:
                    print("\n🤖 DeepSeek Bot: Goodbye! Have a great day! 👋")
                    break
                
                if user_input.lower() == 'clear':
                    self.clear_history()
                    continue
                
                # Get bot response
                print("\n🤖 DeepSeek Bot: Thinking...", end="\r")
                response = self.chat(user_input)
                print(" " * 50, end="\r")  # Clear "Thinking..." message
                
                print(f"🤖 DeepSeek Bot: {response}")
                
            except KeyboardInterrupt:
                print("\n\n🤖 DeepSeek Bot: Goodbye! 👋")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")


# Run the chatbot
if __name__ == "__main__":
    # Option 1: Pass API key directly (not recommended for production)
    # YOUR_API_KEY = "sk_5b9cb3216a0241268097010fc3344d37"  # Replace with your key
    # bot = DeepSeekChatbot(api_key=YOUR_API_KEY)
    
    # Option 2: Use environment variable (RECOMMENDED)
    # Create a .env file with: DEEPSEEK_API_KEY=your_key_here
    bot = DeepSeekChatbot()
    bot.run()

