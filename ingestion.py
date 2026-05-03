import os
import spacy
import pandas as pd
import pdfplumber

from pathlib import Path
from dotenv import load_dotenv
from transformers import AutoTokenizer

load_dotenv()
hf_token = os.getenv('HF_TOKEN')

class KnowledgeIngestor:
    def __init__(
        self,
        # might not init here may be in class instantiation on main.py
        tokenizer = AutoTokenizer.from_pretrained(
            "BAAI/bge-m3",
            token=hf_token,
        ),
        spacy_sentencizer = spacy.load(
            "en_core_web_md",
            exclude = [
                "ner",
                "tagger",
                "lemmatizer",
                "tok2vec",
                "senter",
                "attribute_ruler"
            ]
        ),
        # local dir change to in-function not in init
        local_directory = Path(r"C:\Users\ptrck\flagship\query-agent\documents\Common Workplace IT Support Guide.pdf"),
        pages: int = 0,
        TOKEN_LIMIT: int = 400,
    ):
        self.tokenizer = tokenizer
        self.local_directory = local_directory
        self.pages = pages
        self.TOKEN_LIMIT = TOKEN_LIMIT
        self.spacy_sentencizer = spacy_sentencizer

    def _split_into_sentences(self, text_content: dict) -> list[dict]:
        """
        sentencizer
        input: list of long text
        output list of sentences from long text
        """
        # try:
        # print(text_content)
        doc = self.spacy_sentencizer(text_content.get('text'))
        # sentences = [doc for doc in doc.sents]
        #doc in doc.sents in obj not str

        sentences = []
        for sent in doc.sents:
            current_sentence = {
                "text": sent.text,
                "page_number": text_content.get('page_number'),
                "source_file": text_content.get('source_file')
            }
            sentences.append(current_sentence)

        return sentences

        # except Exception as e:
        #     print(f"Error in sentencizer: {e}")

    def _count_tokens(self, sentence: str) -> int:
        # try:
        token = self.tokenizer(str(sentence))
        token_count = len(token['input_ids'])
        return token_count

        # except Exception as e:
        #     print(f"Error in token counter: {e}")

    def _split_oversized_sentence(self, sentence: str) -> list[str]:
        """
        splits sentences that exceeds te token limit
        returns: correct token count
        TODO: add if the "word" is longer than the TOKEN_LIMIT e.g. urls, logs
        """
        words = sentence.split(" ")
        pieces = []
        current_words = []
        current_tokens = 0

        # try:
        for word in words:
            word_tokens = self._count_tokens(word + " ")

            if current_tokens + word_tokens <= self.TOKEN_LIMIT:
                current_words.append(word)
                current_tokens += word_tokens
            else:
                if current_words:
                    pieces.append(" ".join(current_words))

                current_words = [word]
                current_tokens = word_tokens
        
        if current_words:
            pieces.append(" ".join(current_words))
        
        return pieces

        # except Exception as e:
        #     print(f"Error in splitting the sentences: {e}")

    def _save_chunk(self, text: str, source_file: str, page_start: int, page_end: int, token_count:int) -> dict:

        return {
            "text": text,
            "source_file": source_file,
            "page_start": page_start,
            "page_end": page_end,
            "token_count": token_count
        }


    def _build_chunks(self, sentences: list[dict]) -> list[dict]:
        """
        Tokenizer
        """
        final_chunks = []
        chunks = []
        token_tracker = 0

        # try:

        for sentence in sentences:
            sentence_text = sentence.get('text')
            page_number = sentence.get('page_number')
            sentence_tokens = self._count_tokens(sentence_text)

            if sentence_tokens > self.TOKEN_LIMIT:
                if chunks:
                    # might move to a new method _save_current_chunks
                    chunks_text = " ".join(d["text"] for d in chunks)
                    source_file = chunks[0]["source_file"]
                    page_start = chunks[0]["page_number"]
                    page_end = chunks[-1]["page_number"]
                    token_count  = sum(d["token_count"] for d in chunks)
                    final_chunks_dict = self._save_chunk(
                        chunks_text,
                        source_file,
                        page_start,
                        page_end, 
                        token_count,
                    )
                    final_chunks.append(final_chunks_dict)
                    chunks = []
                    token_tracker = 0
                    
                chunks_text = self._split_oversized_sentence(sentence_text)
                # chunks_text = " ".join(chunks)
                page_start = sentence["page_number"]
                page_end = sentence["page_number"]
                source_file = sentence["source_file"]

                for text in chunks_text:
                    # token_count  = sum(d["token_count"] for d in chunks)
                    token_count = self._count_tokens(text)
                    final_chunks_dict = self._save_chunk(
                        text,
                        source_file,
                        page_start,
                        page_end, 
                        token_count,
                    )
                    final_chunks.append(final_chunks_dict)
            else:                   
                if token_tracker + sentence_tokens <= self.TOKEN_LIMIT:
                    token_tracker += sentence_tokens
                    token_count  = sentence_tokens
                    source_file = sentence.get("source_file")

                    chunks_dict = {
                        "text": sentence_text,
                        "source_file":source_file,
                        "page_number":page_number,
                        "token_count":token_count,
                    }

                    chunks.append(chunks_dict)
                else: 
                    chunks_text = " ".join(d["text"] for d in chunks)
                    source_file = chunks[0]["source_file"]
                    page_start = chunks[0]["page_number"]
                    page_end = chunks[-1]["page_number"]
                    token_count  = sum(d["token_count"] for d in chunks)
                    final_chunks_dict = self._save_chunk(
                        chunks_text,
                        source_file,
                        page_start,
                        page_end, 
                        token_count,
                    )
                    final_chunks.append(final_chunks_dict)
                    # " ".join(chunks))
                    chunks = []
                    token_count = sentence_tokens
                    source_file = sentence.get("source_file")

                    chunks_dict = {
                        "text": sentence_text,
                        "source_file":source_file,
                        "page_number":page_number,
                        "token_count":token_count,
                    }

                    chunks.append(chunks_dict)
                    token_tracker = sentence_tokens

        if chunks:
            chunks_text = " ".join(d["text"] for d in chunks)
            source_file = chunks[0]["source_file"]
            page_start = chunks[0]["page_number"]
            page_end = chunks[-1]["page_number"]
            token_count  = sum(d["token_count"] for d in chunks)
            final_chunks_dict = self._save_chunk(
                chunks_text,
                source_file,
                page_start,
                page_end, 
                token_count,
            )
            final_chunks.append(final_chunks_dict)

        return final_chunks

        # except Exception as e:
        #     print(f"Error failed building chunks: {e}")

    def _build_payloads(self, chunks: list[dict]) -> list[dict]:
        payloads = []

        for index, chunk in enumerate(chunks):

            file_stem = Path(chunk["source_file"]).stem
            file_name = (file_stem.replace(" ", "_")).lower()
            chunk_id = file_name + str("_p" + str(chunk["page_start"]) + "_p" + str(chunk["page_end"]) + "_c" + f"{index:04d}")

            document_type = Path(chunk["source_file"]).suffix.strip(".").lower()
            
            payload = {
                "text": chunk["text"],
                "chunk_id": chunk_id,
                "source_file": chunk["source_file"],
                "page_start": chunk["page_start"],
                "page_end": chunk["page_end"], 
                "token_count": chunk["token_count"],
                "metadata": {
                    "document_type": document_type
                }
            }

            payloads.append(payload)

        return payloads

    def run_pipeline(self) -> list:
        """
        pipeline runner
        output" list of dictionaries
        """
        # try:
        text_content = self._read_pdf()

        sentencizer = [
            sentence
            for text in text_content
            for sentence in self._split_into_sentences(text)
        ]

        chunks = self._build_chunks(sentencizer)
        payloads = self._build_payloads(chunks)

        return payloads

        # except Exception as e:
        #     print(f"Error in pipeline runner occured: {e}")

    def _read_pdf(self) -> list:
        """
        reads pdf content

        input: file
        Output: list of text indexed by page
        """

        # try:
        pdf_content = []
        with pdfplumber.open(self.local_directory) as pdf:
            for page in pdf.pages:
                extracted_text = page.extract_text()
                replace_text = str(extracted_text).replace("\n", " ")

                current_page = {
                    "text": replace_text,
                    "page_number": int(page.page_number),
                    "source_file": self.local_directory.name
                }
                pdf_content.append(current_page)

        return pdf_content

        # except Exception as e:
        #     print(f"Error in reading PDF occured: {e}")

    def _read_csv(self):
        pass

ingestor = KnowledgeIngestor()







