from fastchat.conversation import Conversation, SeparatorStyle, register_conv_template


class ChatTemplates:
    def __init__(self) -> None:
        self.register_templates(self.mistral_teacher_template())
        self.register_templates(self.mistral_answer_predictor())
        self.register_templates(self.mistral_curation_template())
        self.register_templates(self.mistral_frontend_template())
        self.register_templates(self.mistral_mpbb_template()) 
        pass

    def register_templates(self, conv: Conversation):
        register_conv_template(conv, override=True)

    def mistral_teacher_template(self) -> Conversation:
        return Conversation(
            name="mistral-teacher",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            system_message=(
                "You are a teacher for beginners in Python programming to explain Code. "
                "First, explain from which file and module this code snippet is taken and which imports are needed. "
                "Then, explain the code line by line."
            ),
            roles=("[INST]", "[/INST]"),  # type: ignore
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )

    def mistral_answer_predictor(self) -> Conversation:
        return Conversation(
            name="mistral-answer-predictor",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            system_message=(
                "You are a model to generate a question-answer pair. "
                "You will receive an explanation of a code snippet. "
                "The provided function is Python code and is part of the Spyder IDE repository. "
                "Predict a question a user would ask. "
                "Always include the name of the file, the module in the question and the start and end line of the file. "
                "Always include in your answer code from the explanation. "
                "Provide your question-answer pair in the format: "
                "Question: <<Your Question>> "
                "Answer: <<Your Answer>> "
            ),
            roles=("[INST]", "[/INST]"),  # type: ignore
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )

    def mistral_curation_template(self) -> Conversation:
        return Conversation(
            name="mistral-curation",
            system_template="[INST] <<SYS>>\n{system_message}",
            system_message=(
                "Below is an instruction from an user and a candidate answer. Evaluate whether or not the answer is a good example of how AI Assistant should respond to the user's instruction. Please assign a score using the following 5-point scale: "
                "1: It means the answer is incomplete, vague, off-topic, controversial, or not exactly what the user asked for. For example, some content seems missing, numbered list does not start from the beginning, the opening sentence repeats user's question. Or the response is from another person’s perspective with their personal experience (e.g. taken from blog posts), or looks like an answer from a forum. Or it contains promotional text, navigation text, or other irrelevant information. "
                "2: It means the answer addresses most of the asks from the user. It does not directly address the user's question. For example, it only provides a high-level methodology instead of the exact solution to user's question. "
                "3: It means the answer is helpful but not written by an AI Assistant. It addresses all the basic asks from the user. It is complete and self contained with the drawback that the response is not written from an AI assistant's perspective, but from other people's perspective. The content looks like an excerpt from a blog post, web page, or web search results. For example, it contains personal experience or opinion, mentions comments section, or share on social media, etc. "
                "4: It means the answer is written from an AI assistant's perspective with a clear focus of addressing the instruction. It provide a complete, clear, and comprehensive response to user’s question or instruction without missing or irrelevant information. It is well organized, self-contained, and written in a helpful tone. It has minor room for improvement, e.g. more concise and focused. "
                "5: It means it is a perfect answer from an AI Assistant. It has a clear focus on being a helpful AI Assistant, where the response looks like intentionally written to address the user's question or instruction without any irrelevant sentences. The answer provides high quality content, demonstrating expert knowledge in the area, is very well written, logical, easy-to-follow, engaging and insightful. "
                "Please first provide a brief reasoning you used to derive the rating score, and then write 'Score: <rating>' in the last line."
            ),
            roles=("[INST]", "[/INST]"),  # type: ignore
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
    
    def mistral_frontend_template(self) -> Conversation:
        return Conversation(
            name="mistral-frontend",
            system_template="[INST] <<SYS>>\n{system_message}",
            system_message=f"",
            roles=("[INST]", "[/INST]"),  # type: ignore
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )

    def mistral_mpbb_template(self) -> Conversation:
        return Conversation(
            name="mistral-mbpp",
            system_template="[INST] <<SYS>>\n{system_message}\n<</SYS>>\n\n",
            system_message=f"You are an expert Python programmer, and here is your task:\n ",
            roles=("[INST]", "[/INST]"),  # type: ignore
            sep_style=SeparatorStyle.LLAMA2,
            sep=" ",
            sep2=" </s><s>",
        )
