import os
from collections import deque
from typing import List, Iterable

import igraph
import tree_sitter
from hydra import initialize, compose
from loguru import logger
from tree_sitter import Parser
from tree_sitter.binding import Tree, Node

from feature.content import ContentExtraction


class Token(object):
    """
    Container for the token text and its annotations
    """

    def __init__(self, token: str, annotation=None, block_annotation=None, position=0):
        """
        :param token:
        :param annotation:
        :param block_annotation:
        """
        self._token = token
        self._annotation = annotation
        self._block = block_annotation
        self._position = position

    @property
    def token(self):
        return self._token

    @property
    def position(self):
        return self._position

    @property
    def annotation(self):
        return self._annotation

    @property
    def block(self):
        return self._block

    def __str__(self):
        return f"{self.token} - {self.annotation} - {self.block} - {self.position}"


class MethodContentExtraction(ContentExtraction):
    """
    Extracts the features using the identifiers from the source code as representation for the document.
    """

    def __init__(self, graph_path: str = None,
                 repo_path: str = None, stopwords: Iterable = None):
        super().__init__(graph_path, stopwords)

        # Dirty fix for loading the tree-sitter language file
        with initialize(version_base=None, config_path="../../src/conf/"):
            cfg = compose(config_name='annotation.yaml')
        lang = tree_sitter.Language(f'{cfg.base_path}/languages.so', 'java')
        self.parser = Parser()
        self.parser.set_language(lang)

        self.repositories = repo_path
        self.method = 'methods'

    def get_content(self, project: str, graph: igraph.Graph):
        """
        Returns the embeddings of files in the project.
        :param project: Name of the project
        :param graph: Graph of the project
        :return:
        """
        for node in graph.vs:
            path = os.path.join(self.repositories, project, node['filePathRelative'])
            if not os.path.isfile(path):
                continue

            methods = self.get_methods(path)
            yield node['filePathRelative'], methods

    @staticmethod
    def read_file(filename: str):
        """
        Reads the file and returns the text.
        :param filename:
        :return:
        """
        try:
            with open(filename, "rb") as inf:
                text = inf.read()
        except Exception as e:
            logger.error(f"Error reading file {filename}: {e}")
            return ""

        return text

    def get_methods(self, path: str):
        """
        Returns the source code identifiers from the file.
        :param path:
        :return:
        """
        file_content = self.read_file(path)
        tree = self.parser.parse(file_content)
        tokens = self._traverse(file_content, tree)
        methods = []
        method = {}
        i = 0
        while len(tokens) > i + 1:
            if tokens[i].annotation == 'identifier' and tokens[i].block == 'method_declaration':
                start = tokens[i].position
                method['name'] = tokens[i].token
                in_method = True
                open_blocks = 0
                closed_blocks = 0
                comments = [start]
                while in_method and len(tokens) > i + 1:
                    i += 1
                    if 'comment' in tokens[i].annotation:
                        comments.extend([tokens[i].position, tokens[i + 1].position])
                    if tokens[i].block == 'block' and tokens[i].annotation == '{':
                        open_blocks += 1
                    if tokens[i].block == 'block' and tokens[i].annotation == '}':
                        closed_blocks += 1

                    if open_blocks != 0 and open_blocks == closed_blocks:
                        in_method = False
                        i += 1

                end = tokens[i].position
                comments.append(end)
                pairs = list(zip(comments[::2], comments[1::2]))
                body = [file_content[s:e] for s, e in pairs]
                body = ' '.join([t.decode("utf8") for t in body])
                method['body'] = ' '.join(body.split())
                methods.append(method)
            method = {}
            i += 1

        return methods

    @staticmethod
    def _extract_token_annotation(code: bytes, node: Node) -> (bytes, str):
        """
        Extract the token string from the code
        :param code: The code textual representation
        :param node: A node containing the start and end positions of the token
        :return:
        """
        token = code[node.start_byte:node.end_byte]
        annotations = node.type
        return token, annotations

    def _traverse(self, code: bytes, tree: Tree) -> List[Token]:
        """
        Post-order tree traversal that returns a list of tokens with their annotations
        :param code: A byte string representation of the code
        :param tree: The tree representation of the code
        :return:
        """
        root = tree.root_node
        stack = deque()
        stack.append((root, None))

        tokens = []
        while len(stack):
            current, parent = stack.pop()

            if current.type != tree.root_node.type and len(current.children) == 0:
                token, annotation = self._extract_token_annotation(code, current)
                _, block_annotation = self._extract_token_annotation(code, parent)
                tokens.append(Token(token.decode("utf8"), annotation, block_annotation, current.start_byte))
            for child in current.children:
                stack.append((child, current))

        return tokens[::-1]
