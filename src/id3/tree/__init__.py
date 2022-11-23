class Node:
    def __init__(self, data):
        self.left = None
        self.right = None
        self.data = data


class Tree:
    root: Node = None

    def __init__(self, root_data):
        self.root = Node(root_data)
