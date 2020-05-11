"""
Date: 2020/05/11
Code for Huffman Coding, compression and decompression.
Modified based on Bhirgu's code.
Original code can be found at: https://github.com/bhrigu123/huffman-coding/blob/master/huffman.py
"""

import heapq
import os
from functools import total_ordering
from easydict import EasyDict


@total_ordering
class HeapNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None

    # defining comparators less_than and equals
    def __lt__(self, other):
        return self.freq < other.freq

    def __eq__(self, other):
        if other is None:
            return False
        if not isinstance(other, HeapNode):
            return False
        return self.freq == other.freq


class HuffmanCoding:
    def __init__(self, data):
        self.data = data
        self.freq_dict = self.make_frequency_dict()
        self.heap = self.make_heap()
        self.codes = {}
        self.reverse_mapping = {}
        self.merge_nodes()
        self.make_codes()
        self.encoded_text = []
        self.padded_encoded_text = []

    # functions for compression:

    def make_frequency_dict(self):
        # Performed at __init__ for better initialization.
        frequency = {}
        for character in self.data:
            if character not in frequency:
                frequency[character] = 0
            frequency[character] += 1
        return frequency

    def make_heap(self):
        heap = []
        for key in self.freq_dict:
            node = HeapNode(key, self.freq_dict[key])
            heapq.heappush(heap, node)
        return heap

    def merge_nodes(self):
        while len(self.heap) > 1:
            node1 = heapq.heappop(self.heap)
            node2 = heapq.heappop(self.heap)

            merged = HeapNode(None, node1.freq + node2.freq)
            merged.left = node1
            merged.right = node2

            heapq.heappush(self.heap, merged)

    def make_codes_helper(self, root, current_code):
        if root is None:
            return

        if root.char is not None:
            # Pay attention to here.
            self.codes[root.char] = current_code
            self.reverse_mapping[current_code] = root.char
            return

        self.make_codes_helper(root.left, current_code + "0")
        self.make_codes_helper(root.right, current_code + "1")

    def make_codes(self):
        root = heapq.heappop(self.heap)
        current_code = ""
        self.make_codes_helper(root, current_code)

    def get_encoded_text(self):
        encoded_text = ""
        for character in self.data:
            encoded_text += self.codes[character]
        return encoded_text

    def pad_encoded_text(self):
        encoded_text = self.encoded_text
        extra_padding = 8 - len(encoded_text) % 8
        for i in range(extra_padding):
            encoded_text += "0"

        padded_info = "{0:08b}".format(extra_padding)
        encoded_text += padded_info
        return encoded_text

    def get_byte_array(self):
        padded_encoded_text = self.padded_encoded_text
        b = bytearray()
        if len(padded_encoded_text) % 8 != 0:
            raise ValueError("Encoded text not padded properly")

        for i in range(0, len(padded_encoded_text), 8):
            byte = padded_encoded_text[i:i + 8]
            b.append(int(byte, 2))
        return b

    def compression(self):
        self.encoded_text = self.get_encoded_text()
        print('encoded string length: {}'.format(len(self.encoded_text)))
        self.padded_encoded_text = self.pad_encoded_text()

        byte_stream = bytes(self.get_byte_array())
        return byte_stream

    # def compress(self):
    #     filename, file_extension = os.path.splitext(self.data)
    #     output_path = filename + ".bin"
    #
    #     with open(self.data, 'r+') as file, open(output_path, 'wb') as output:
    #         text = file.read()
    #         text = text.rstrip()
    #
    #         frequency = self.make_frequency_dict(text)
    #         self.make_heap(frequency)
    #         self.merge_nodes()
    #         self.make_codes()
    #
    #         encoded_text = self.get_encoded_text(text)
    #         padded_encoded_text = self.pad_encoded_text(encoded_text)
    #
    #         b = self.get_byte_array(padded_encoded_text)
    #         output.write(bytes(b))
    #
    #     print("Compressed")
    #     return output_path

    """ functions for decompression: """

    def remove_padding(self, padded_encoded_text):
        padded_info = padded_encoded_text[:8]
        extra_padding = int(padded_info, 2)

        padded_encoded_text = padded_encoded_text[8:]
        encoded_text = padded_encoded_text[:-1 * extra_padding]

        return encoded_text

    def decode_text(self, encoded_text):
        current_code = ""
        decoded_text = ""

        for bit in encoded_text:
            current_code += bit
            if current_code in self.reverse_mapping:
                character = self.reverse_mapping[current_code]
                decoded_text += character
                current_code = ""

        return decoded_text

    def decompress(self, input_path):
        filename, file_extension = os.path.splitext(self.data)
        output_path = filename + "_decompressed" + ".txt"

        with open(input_path, 'rb') as file, open(output_path, 'w') as output:
            bit_string = ""

            byte = file.read(1)
            while len(byte) > 0:
                byte = ord(byte)
                bits = bin(byte)[2:].rjust(8, '0')
                bit_string += bits
                byte = file.read(1)

            encoded_text = self.remove_padding(bit_string)

            decompressed_text = self.decode_text(encoded_text)

            output.write(decompressed_text)

        print("Decompressed")
        return output_path
