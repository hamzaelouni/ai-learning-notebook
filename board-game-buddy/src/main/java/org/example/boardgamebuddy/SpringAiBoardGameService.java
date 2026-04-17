package org.example.boardgamebuddy;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.stereotype.Service;

@Service
public class SpringAiBoardGameService implements BoardGameService{

    private final ChatClient chatClient;

    public SpringAiBoardGameService(ChatClient.Builder chatClientBuilder) {
        this.chatClient = chatClientBuilder.build();
    }

    @Override
    public Answer askquestion(Question question) {
        var answerText = chatClient.prompt()
                //user() is a method that defines a message in the prompt for the "user" role
                //system() is a method that defines a message in the prompt for the "system" role
                .user(question.question())
                .call()
                .content();
        return new Answer(answerText);
    }
}
