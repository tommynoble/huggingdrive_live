#!/usr/bin/env python3
"""
Gradio Interface Manager for HuggingDrive
Handles creation of web interfaces for different model types
"""

import os
import json
import threading
import webbrowser
from pathlib import Path
from typing import Dict, List, Optional, Any
import gradio as gr
from transformers.pipelines import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import torch


class GradioInterfaceManager:
    """Manages Gradio web interfaces for HuggingDrive models"""

    def __init__(self):
        self.active_interfaces = {}  # model_name -> interface
        self.model_pipelines = {}  # model_name -> pipeline
        self.interface_threads = {}  # model_name -> thread

    def detect_model_type(self, model_path: str) -> str:
        """Detect the type of model based on config.json"""
        try:
            config_path = Path(model_path) / "config.json"
            if not config_path.exists():
                return "unknown"

            with open(config_path, "r") as f:
                config = json.load(f)

            model_type = config.get("model_type", "unknown")
            arch = config.get("architectures", [])

            # Determine specific task type
            if model_type in [
                "gpt2",
                "gpt",
                "causal_lm",
                "llama",
                "mistral",
                "falcon",
                "bloom",
                "opt",
            ]:
                return "text-generation"
            elif model_type in ["bert", "roberta", "distilbert"]:
                return "text-classification"
            elif model_type in ["t5", "bart", "marian"]:
                return "translation"
            elif "vision" in model_type or "image" in model_type:
                return "image-generation"
            elif "audio" in model_type or "speech" in model_type:
                return "audio-processing"
            else:
                return "text-generation"  # Default fallback

        except Exception as e:
            print(f"Error detecting model type: {e}")
            return "text-generation"

    def create_text_generation_interface(self, model_path: str, model_name: str):
        """Create Gradio interface for text generation models"""

        def generate_text(prompt, max_length, temperature, top_p, do_sample):
            try:
                if model_name not in self.model_pipelines:
                    # Load model if not already loaded
                    pipeline_obj = pipeline("text-generation", model=model_path)
                    self.model_pipelines[model_name] = pipeline_obj

                pipeline_obj = self.model_pipelines[model_name]

                # Generate text
                outputs = pipeline_obj(
                    prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=pipeline_obj.tokenizer.eos_token_id,
                    return_full_text=False,
                )

                return outputs[0]["generated_text"]
            except Exception as e:
                return f"Error generating text: {str(e)}"

        # Create interface
        with gr.Blocks(title=f"Text Generation - {model_name}") as interface:
            gr.Markdown(f"# Text Generation with {model_name}")
            gr.Markdown("Generate text continuations using the loaded model.")

            with gr.Row():
                with gr.Column(scale=2):
                    input_text = gr.Textbox(
                        label="Input Text",
                        placeholder="Enter your prompt here...",
                        lines=5,
                    )
                    generate_btn = gr.Button("Generate", variant="primary")
                    output_text = gr.Textbox(
                        label="Generated Text", lines=10, interactive=False
                    )

                with gr.Column(scale=1):
                    gr.Markdown("### Generation Parameters")
                    max_length = gr.Slider(
                        minimum=10, maximum=500, value=100, step=10, label="Max Length"
                    )
                    temperature = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature",
                    )
                    top_p = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.1,
                        label="Top-p (nucleus sampling)",
                    )
                    do_sample = gr.Checkbox(label="Use sampling", value=True)

            generate_btn.click(
                fn=generate_text,
                inputs=[input_text, max_length, temperature, top_p, do_sample],
                outputs=output_text,
            )

        return interface

    def create_chat_interface(self, model_path: str, model_name: str):
        """Create Gradio interface for chat models"""

        def chat_with_model(message, history, max_length, temperature):
            try:
                if model_name not in self.model_pipelines:
                    # Load model if not already loaded
                    pipeline_obj = pipeline("text-generation", model=model_path)
                    self.model_pipelines[model_name] = pipeline_obj

                pipeline_obj = self.model_pipelines[model_name]

                # Build conversation context
                conversation = ""
                for human, assistant in history:
                    conversation += f"Human: {human}\nAssistant: {assistant}\n"
                conversation += f"Human: {message}\nAssistant:"

                # Generate response
                outputs = pipeline_obj(
                    conversation,
                    max_length=len(pipeline_obj.tokenizer.encode(conversation))
                    + max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=pipeline_obj.tokenizer.eos_token_id,
                    return_full_text=False,
                )

                response = outputs[0]["generated_text"].strip()

                # Return updated history (Gradio chatbot expects this format)
                history.append((message, response))
                return history
            except Exception as e:
                error_msg = f"Error in chat: {str(e)}"
                history.append((message, error_msg))
                return history

        # Create interface
        with gr.Blocks(title=f"Chat - {model_name}") as interface:
            gr.Markdown(f"# Chat with {model_name}")
            gr.Markdown("Have a conversation with the AI model.")

            chatbot = gr.Chatbot(height=400)
            msg = gr.Textbox(
                label="Your message", placeholder="Type your message here..."
            )
            clear = gr.Button("Clear")

            with gr.Row():
                max_length = gr.Slider(
                    minimum=50,
                    maximum=200,
                    value=100,
                    step=10,
                    label="Max Response Length",
                )
                temperature = gr.Slider(
                    minimum=0.1, maximum=2.0, value=0.7, step=0.1, label="Temperature"
                )

            msg.submit(
                fn=chat_with_model,
                inputs=[msg, chatbot, max_length, temperature],
                outputs=[chatbot],
            )

            clear.click(lambda: None, None, chatbot, queue=False)

        return interface

    def create_unified_interface(
        self, model_path: str, model_name: str, model_type: str
    ):
        """Create a unified Gradio interface with tabs for different capabilities"""

        with gr.Blocks(title=f"{model_name} - Web Interface") as interface:
            gr.Markdown(f"# 🤖 {model_name}")
            gr.Markdown(f"**Model Type:** {model_type.replace('-', ' ').title()}")
            gr.Markdown("Choose a tab below to interact with your model:")

            # Model status indicator
            def get_model_status():
                if model_name in self.model_pipelines:
                    return "🟢 Model Loaded", "Model is loaded and ready to use"
                else:
                    return (
                        "🔴 Model Not Loaded",
                        "Model will be loaded automatically when you start using it",
                    )

            def unload_model_from_gradio():
                """Unload model from Gradio interface"""
                if model_name in self.model_pipelines:
                    try:
                        pipeline_obj = self.model_pipelines[model_name]
                        del pipeline_obj
                        import gc

                        gc.collect()
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except:
                            pass
                        del self.model_pipelines[model_name]
                        return (
                            "🔴 Model Unloaded",
                            "Model has been unloaded from memory",
                        )
                    except Exception as e:
                        return "⚠️ Error", f"Error unloading model: {str(e)}"
                else:
                    return "🔴 Model Not Loaded", "Model is not currently loaded"

            status_text, status_desc = get_model_status()
            status_display = gr.Markdown(f"**{status_text}** - {status_desc}")

            # Add unload button
            unload_btn = gr.Button("🗑️ Unload Model", variant="secondary", size="sm")
            unload_btn.click(fn=unload_model_from_gradio, outputs=[status_display])

            with gr.Tabs():
                # Chat Tab (for conversational models)
                with gr.Tab(
                    "💬 Chat", visible=model_type in ["text-generation", "chat"]
                ):
                    gr.Markdown("### Have a conversation with your AI model")
                    gr.Markdown(
                        "💡 **Tips:** Start with simple greetings like 'Hello!' or ask questions like 'What can you help me with?' The model works best with clear, specific requests."
                    )
                    chatbot = gr.Chatbot(height=400, label="Chat History")
                    with gr.Row():
                        msg = gr.Textbox(
                            label="Your message",
                            placeholder="Try: 'Hello!', 'What can you help me with?', or 'Tell me a joke'",
                            scale=4,
                        )
                        send_btn = gr.Button("Send", variant="primary", scale=1)
                    with gr.Row():
                        clear = gr.Button("🗑️ Clear Chat", variant="secondary")
                        export_btn = gr.Button("📄 Export Chat", variant="secondary")

                    with gr.Row():
                        max_length = gr.Slider(
                            minimum=50,
                            maximum=300,
                            value=150,
                            step=10,
                            label="Max Response Length",
                        )
                        temperature = gr.Slider(
                            minimum=0.1,
                            maximum=2.0,
                            value=0.8,
                            step=0.1,
                            label="Temperature (0.8 = balanced, 1.2 = creative)",
                        )

                    def chat_with_model(message, history, max_length, temperature):
                        try:
                            if model_name not in self.model_pipelines:
                                pipeline_obj = pipeline(
                                    "text-generation", model=model_path
                                )
                                self.model_pipelines[model_name] = pipeline_obj

                            pipeline_obj = self.model_pipelines[model_name]

                            # Ensure history is a list of tuples
                            if history is None:
                                history = []
                            elif not isinstance(history, list):
                                history = []

                            # Create a copy of history to avoid modifying the original
                            history_copy = history.copy()

                            # Build conversation context for TinyLlama and similar models
                            # Try different formats based on model capabilities
                            conversation = ""

                            # Use a simpler, more reliable format for TinyLlama
                            # TinyLlama works better with simple instruction format
                            if not history_copy:
                                # First message - use instruction format
                                conversation = f"Below is a conversation between a helpful AI assistant and a user.\n\nUser: {message}\nAssistant:"
                            else:
                                # Build conversation context
                                conversation = "Below is a conversation between a helpful AI assistant and a user.\n\n"
                                for human, assistant in history_copy:
                                    conversation += (
                                        f"User: {human}\nAssistant: {assistant}\n"
                                    )
                                conversation += f"User: {message}\nAssistant:"

                            # Generate response with optimized parameters for TinyLlama
                            outputs = pipeline_obj(
                                conversation,
                                max_new_tokens=max_length,
                                temperature=temperature,
                                do_sample=True,
                                top_p=0.9,
                                repetition_penalty=1.1,
                                pad_token_id=pipeline_obj.tokenizer.eos_token_id,
                                eos_token_id=pipeline_obj.tokenizer.eos_token_id,
                                return_full_text=False,
                                truncation=True,
                            )

                            response = outputs[0]["generated_text"].strip()

                            # Clean up response - remove any remaining tokens or prefixes
                            if "User:" in response:
                                response = response.split("User:")[0].strip()
                            if "Human:" in response:
                                response = response.split("Human:")[0].strip()
                            if "<|user|>" in response:
                                response = response.split("<|user|>")[0].strip()
                            if "<|system|>" in response:
                                response = response.split("<|system|>")[0].strip()
                            if "<|im_start|>" in response:
                                response = response.split("<|im_start|>")[0].strip()
                            if "<|im_end|>" in response:
                                response = response.split("<|im_end|>")[0].strip()

                            # Ensure we have a valid response
                            if not response:
                                response = "I'm sorry, I couldn't generate a response. Please try again."

                            # Return new history with the message and response
                            new_history = history_copy + [(message, response)]
                            # Debug: ensure we're returning the correct format
                            if not isinstance(new_history, list):
                                new_history = history_copy + [
                                    (message, "Error: Invalid history format")
                                ]
                            print(
                                f"DEBUG: Returning history with {len(new_history)} items"
                            )
                            return new_history

                        except Exception as e:
                            error_msg = f"Error in chat: {str(e)}"
                            # Return new history with error message
                            history_copy = history.copy() if history else []
                            return history_copy + [(message, error_msg)]

                    # Wrapper function to clear input after chat
                    def chat_and_clear(message, history, max_length, temperature):
                        if not message.strip():  # Don't process empty messages
                            return history, ""
                        try:
                            result = chat_with_model(
                                message, history, max_length, temperature
                            )
                            # Ensure result is a list of tuples
                            if not isinstance(result, list):
                                result = history + [
                                    (message, "Error: Invalid response format")
                                ]
                            return result, ""  # Return empty string to clear input
                        except Exception as e:
                            # Return error in correct format
                            error_history = history + [(message, f"Error: {str(e)}")]
                            return error_history, ""

                    msg.submit(
                        fn=chat_and_clear,
                        inputs=[msg, chatbot, max_length, temperature],
                        outputs=[chatbot, msg],  # Also output to msg to clear it
                        show_progress="hidden",  # Hide the loader
                    )

                    send_btn.click(
                        fn=chat_and_clear,
                        inputs=[msg, chatbot, max_length, temperature],
                        outputs=[chatbot, msg],  # Also output to msg to clear it
                        show_progress="hidden",  # Hide the loader
                    )

                    clear.click(lambda: None, None, chatbot, queue=False)

                    def export_chat(history):
                        if not history:
                            return "No chat history to export."

                        export_text = f"Chat Export - {model_name}\n"
                        export_text += "=" * 50 + "\n\n"

                        for i, (user_msg, assistant_msg) in enumerate(history, 1):
                            export_text += f"Exchange {i}:\n"
                            export_text += f"User: {user_msg}\n"
                            export_text += f"Assistant: {assistant_msg}\n"
                            export_text += "-" * 30 + "\n\n"

                        return export_text

                    export_btn.click(
                        fn=export_chat,
                        inputs=[chatbot],
                        outputs=[msg],  # Use the message input to show export result
                    )

                # Text Generation Tab
                with gr.Tab(
                    "✍️ Text Generation",
                    visible=model_type in ["text-generation", "chat"],
                ):
                    gr.Markdown("### Generate text continuations")
                    gr.Markdown(
                        "💡 **Tips:** Use presets for quick setup, or fine-tune parameters for custom results. Higher temperature = more creative, lower = more focused."
                    )

                    with gr.Row():
                        with gr.Column(scale=2):
                            input_text = gr.Textbox(
                                label="Input Text",
                                placeholder="Enter your prompt here...",
                                lines=5,
                            )
                            generate_btn = gr.Button("Generate", variant="primary")
                            output_text = gr.Textbox(
                                label="Generated Text", lines=10, interactive=False
                            )

                        with gr.Column(scale=1):
                            gr.Markdown("### Generation Parameters")

                            # Basic parameters
                            max_length = gr.Slider(
                                minimum=10,
                                maximum=1000,
                                value=150,
                                step=10,
                                label="Max New Tokens",
                            )
                            temperature = gr.Slider(
                                minimum=0.1,
                                maximum=2.0,
                                value=0.8,
                                step=0.1,
                                label="Temperature (0.8 = balanced)",
                            )

                            # Sampling parameters
                            top_p = gr.Slider(
                                minimum=0.1,
                                maximum=1.0,
                                value=0.9,
                                step=0.05,
                                label="Top-p (nucleus sampling)",
                            )
                            top_k = gr.Slider(
                                minimum=1, maximum=100, value=50, step=1, label="Top-k"
                            )

                            # Repetition control
                            repetition_penalty = gr.Slider(
                                minimum=0.8,
                                maximum=2.0,
                                value=1.1,
                                step=0.1,
                                label="Repetition Penalty",
                            )
                            length_penalty = gr.Slider(
                                minimum=0.5,
                                maximum=2.0,
                                value=1.0,
                                step=0.1,
                                label="Length Penalty",
                            )

                            # Advanced options
                            do_sample = gr.Checkbox(label="Use sampling", value=True)
                            early_stopping = gr.Checkbox(
                                label="Early stopping", value=True
                            )
                            num_beams = gr.Slider(
                                minimum=1,
                                maximum=10,
                                value=1,
                                step=1,
                                label="Beam Search (1 = greedy)",
                            )

                            # Preset buttons
                            with gr.Row():
                                creative_btn = gr.Button("🎨 Creative", size="sm")
                                focused_btn = gr.Button("🎯 Focused", size="sm")
                                balanced_btn = gr.Button("⚖️ Balanced", size="sm")

                    def generate_text(
                        prompt,
                        max_length,
                        temperature,
                        top_p,
                        top_k,
                        repetition_penalty,
                        length_penalty,
                        do_sample,
                        early_stopping,
                        num_beams,
                    ):
                        try:
                            if model_name not in self.model_pipelines:
                                pipeline_obj = pipeline(
                                    "text-generation", model=model_path
                                )
                                self.model_pipelines[model_name] = pipeline_obj

                            pipeline_obj = self.model_pipelines[model_name]

                            # Build generation parameters
                            gen_params = {
                                "max_new_tokens": max_length,
                                "temperature": temperature,
                                "top_p": top_p,
                                "top_k": top_k,
                                "repetition_penalty": repetition_penalty,
                                "length_penalty": length_penalty,
                                "do_sample": do_sample,
                                "early_stopping": early_stopping,
                                "pad_token_id": pipeline_obj.tokenizer.eos_token_id,
                                "eos_token_id": pipeline_obj.tokenizer.eos_token_id,
                                "return_full_text": False,
                            }

                            # Add beam search if num_beams > 1
                            if num_beams > 1:
                                gen_params["num_beams"] = num_beams
                                gen_params["do_sample"] = (
                                    False  # Beam search doesn't use sampling
                                )

                            outputs = pipeline_obj(prompt, **gen_params)

                            return outputs[0]["generated_text"]
                        except Exception as e:
                            return f"Error generating text: {str(e)}"

                    generate_btn.click(
                        fn=generate_text,
                        inputs=[
                            input_text,
                            max_length,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                            length_penalty,
                            do_sample,
                            early_stopping,
                            num_beams,
                        ],
                        outputs=output_text,
                    )

                    # Preset functions
                    def set_creative():
                        return 200, 1.2, 0.9, 40, 1.0, 1.0, True, True, 1

                    def set_focused():
                        return 100, 0.3, 0.8, 20, 1.2, 1.2, True, True, 1

                    def set_balanced():
                        return 150, 0.8, 0.9, 50, 1.1, 1.0, True, True, 1

                    creative_btn.click(
                        fn=set_creative,
                        outputs=[
                            max_length,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                            length_penalty,
                            do_sample,
                            early_stopping,
                            num_beams,
                        ],
                    )

                    focused_btn.click(
                        fn=set_focused,
                        outputs=[
                            max_length,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                            length_penalty,
                            do_sample,
                            early_stopping,
                            num_beams,
                        ],
                    )

                    balanced_btn.click(
                        fn=set_balanced,
                        outputs=[
                            max_length,
                            temperature,
                            top_p,
                            top_k,
                            repetition_penalty,
                            length_penalty,
                            do_sample,
                            early_stopping,
                            num_beams,
                        ],
                    )

                # Classification Tab
                with gr.Tab(
                    "🏷️ Text Classification", visible=model_type == "text-classification"
                ):
                    gr.Markdown("### Classify text into categories")

                    with gr.Row():
                        with gr.Column():
                            input_text = gr.Textbox(
                                label="Text to Classify",
                                placeholder="Enter text to classify...",
                                lines=5,
                            )
                            classify_btn = gr.Button("Classify", variant="primary")

                        with gr.Column():
                            output_text = gr.Markdown(label="Classification Results")

                    def classify_text(text):
                        try:
                            if model_name not in self.model_pipelines:
                                pipeline_obj = pipeline(
                                    "text-classification", model=model_path
                                )
                                self.model_pipelines[model_name] = pipeline_obj

                            pipeline_obj = self.model_pipelines[model_name]
                            outputs = pipeline_obj(text)

                            if isinstance(outputs, list):
                                results = []
                                for output in outputs:
                                    results.append(
                                        f"**{output['label']}**: {output['score']:.3f}"
                                    )
                                return "\n".join(results)
                            else:
                                return f"**{outputs['label']}**: {outputs['score']:.3f}"
                        except Exception as e:
                            return f"Error classifying: {str(e)}"

                    classify_btn.click(
                        fn=classify_text, inputs=input_text, outputs=output_text
                    )

                # Translation Tab
                with gr.Tab("🌐 Translation", visible=model_type == "translation"):
                    gr.Markdown("### Translate text between languages")

                    with gr.Row():
                        with gr.Column():
                            input_text = gr.Textbox(
                                label="Text to Translate",
                                placeholder="Enter text to translate...",
                                lines=5,
                            )
                            max_length = gr.Slider(
                                minimum=50,
                                maximum=200,
                                value=100,
                                step=10,
                                label="Max Length",
                            )
                            translate_btn = gr.Button("Translate", variant="primary")

                        with gr.Column():
                            output_text = gr.Textbox(
                                label="Translation", lines=5, interactive=False
                            )

                    def translate_text(text, max_length):
                        try:
                            if model_name not in self.model_pipelines:
                                pipeline_obj = pipeline("translation", model=model_path)
                                self.model_pipelines[model_name] = pipeline_obj

                            pipeline_obj = self.model_pipelines[model_name]
                            outputs = pipeline_obj(text, max_length=max_length)
                            return outputs[0]["translation_text"]
                        except Exception as e:
                            return f"Error translating: {str(e)}"

                    translate_btn.click(
                        fn=translate_text,
                        inputs=[input_text, max_length],
                        outputs=output_text,
                    )

                # Model Info Tab
                with gr.Tab("ℹ️ Model Info"):
                    gr.Markdown("### Model Information")
                    gr.Markdown(
                        f"""
                    **Model Name:** {model_name}
                    
                    **Model Type:** {model_type.replace('-', ' ').title()}
                    
                    **Model Path:** {model_path}
                    
                    **Available Features:**
                    - 💬 Chat: {'✅ Available' if model_type in ['text-generation', 'chat'] else '❌ Not available'}
                    - ✍️ Text Generation: {'✅ Available' if model_type in ['text-generation', 'chat'] else '❌ Not available'}
                    - 🏷️ Classification: {'✅ Available' if model_type == 'text-classification' else '❌ Not available'}
                    - 🌐 Translation: {'✅ Available' if model_type == 'translation' else '❌ Not available'}
                    
                    **Usage Tips:**
                    - Use the Chat tab for conversational interactions
                    - Use Text Generation for creative writing and continuations
                    - Adjust temperature and other parameters to control output creativity
                    - Higher temperature = more creative, Lower temperature = more focused
                    """
                    )

        return interface

    def create_translation_interface(self, model_path: str, model_name: str):
        """Create Gradio interface for translation models"""

        def translate_text(text, max_length):
            try:
                if model_name not in self.model_pipelines:
                    # Load model if not already loaded
                    pipeline_obj = pipeline("translation", model=model_path)
                    self.model_pipelines[model_name] = pipeline_obj

                pipeline_obj = self.model_pipelines[model_name]

                # Translate
                outputs = pipeline_obj(text, max_length=max_length)
                return outputs[0]["translation_text"]
            except Exception as e:
                return f"Error translating: {str(e)}"

        # Create interface
        with gr.Blocks(title=f"Translation - {model_name}") as interface:
            gr.Markdown(f"# Translation with {model_name}")
            gr.Markdown("Translate text using the loaded model.")

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Text to Translate",
                        placeholder="Enter text to translate...",
                        lines=5,
                    )
                    max_length = gr.Slider(
                        minimum=50, maximum=200, value=100, step=10, label="Max Length"
                    )
                    translate_btn = gr.Button("Translate", variant="primary")

                with gr.Column():
                    output_text = gr.Textbox(
                        label="Translation", lines=5, interactive=False
                    )

            translate_btn.click(
                fn=translate_text, inputs=[input_text, max_length], outputs=output_text
            )

        return interface

    def create_classification_interface(self, model_path: str, model_name: str):
        """Create Gradio interface for text classification models"""

        def classify_text(text):
            try:
                if model_name not in self.model_pipelines:
                    # Load model if not already loaded
                    pipeline_obj = pipeline("text-classification", model=model_path)
                    self.model_pipelines[model_name] = pipeline_obj

                pipeline_obj = self.model_pipelines[model_name]

                # Classify
                outputs = pipeline_obj(text)

                # Format results
                if isinstance(outputs, list):
                    results = []
                    for output in outputs:
                        results.append(f"**{output['label']}**: {output['score']:.3f}")
                    return "\n".join(results)
                else:
                    return f"**{outputs['label']}**: {outputs['score']:.3f}"
            except Exception as e:
                return f"Error classifying: {str(e)}"

        # Create interface
        with gr.Blocks(title=f"Text Classification - {model_name}") as interface:
            gr.Markdown(f"# Text Classification with {model_name}")
            gr.Markdown("Classify text using the loaded model.")

            with gr.Row():
                with gr.Column():
                    input_text = gr.Textbox(
                        label="Text to Classify",
                        placeholder="Enter text to classify...",
                        lines=5,
                    )
                    classify_btn = gr.Button("Classify", variant="primary")

                with gr.Column():
                    output_text = gr.Markdown(label="Classification Results")

            classify_btn.click(fn=classify_text, inputs=input_text, outputs=output_text)

        return interface

    def launch_interface(
        self, model_path: str, model_name: str, port: int = 7860
    ) -> str:
        """Launch a Gradio interface for the specified model"""

        # Detect model type
        model_type = self.detect_model_type(model_path)

        # Create a unified interface with tabs for different capabilities
        interface = self.create_unified_interface(model_path, model_name, model_type)

        # Find available port
        available_port = self._find_available_port(port)

        # Launch interface in a separate thread
        def launch_gradio():
            interface.launch(
                server_name="0.0.0.0",
                server_port=available_port,
                share=False,
                quiet=True,
            )

        thread = threading.Thread(target=launch_gradio, daemon=True)
        thread.start()

        # Store interface info
        self.active_interfaces[model_name] = interface
        self.interface_threads[model_name] = thread

        # Open browser
        url = f"http://localhost:{available_port}"
        webbrowser.open(url)

        return url

    def _find_available_port(self, start_port: int) -> int:
        """Find an available port starting from start_port"""
        import socket

        port = start_port
        while port < start_port + 100:  # Try up to 100 ports
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.bind(("localhost", port))
                    return port
            except OSError:
                port += 1

        return start_port  # Fallback

    def stop_interface(self, model_name: str):
        """Stop the interface for a specific model"""
        if model_name in self.active_interfaces:
            try:
                # Try to close the Gradio interface gracefully
                interface = self.active_interfaces[model_name]
                if hasattr(interface, "close"):
                    interface.close()
            except Exception as e:
                print(f"Warning: Could not close interface for {model_name}: {e}")

            # Clean up model pipeline and free memory
            if model_name in self.model_pipelines:
                try:
                    pipeline_obj = self.model_pipelines[model_name]
                    # Clear the pipeline object
                    del pipeline_obj
                    # Force garbage collection
                    import gc

                    gc.collect()
                    # Clear CUDA cache if available
                    try:
                        import torch

                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                    except:
                        pass
                    print(f"Model {model_name} unloaded from Gradio interface")
                except Exception as e:
                    print(f"Warning: Could not properly unload model {model_name}: {e}")
                finally:
                    del self.model_pipelines[model_name]

            # Clean up references
            del self.active_interfaces[model_name]
            if model_name in self.interface_threads:
                del self.interface_threads[model_name]

    def stop_all_interfaces(self):
        """Stop all active interfaces - call this when shutting down the app"""
        model_names = list(self.active_interfaces.keys())
        for model_name in model_names:
            self.stop_interface(model_name)
        print("All Gradio interfaces stopped.")

    def get_active_interfaces(self) -> List[str]:
        """Get list of active interface model names"""
        return list(self.active_interfaces.keys())

    def is_model_loaded(self, model_name: str) -> bool:
        """Check if a model is loaded in the Gradio interface"""
        return model_name in self.model_pipelines

    def get_loaded_models(self) -> List[str]:
        """Get list of models currently loaded in Gradio interfaces"""
        return list(self.model_pipelines.keys())
