class Graph:
    """Implementation of an undirected graph, based on Pygraph"""
    weight_attribute_name = "weight"
    default_weight = 0
    label_attrbute_name = "label"
    default_label = ""

    def __init__(self):
        # 关于边的原数据
        self.edge_properties = {}  # Mapping: Edge-> Dict mapping, label->str, word->num
        self.edge_attr = {}  # key value pairs: (Edge->Attributes)
        # 关于节点的原数据
        self.node_attr = {}  # pairing: Node->Attributes
        self.node_neighbors = {}  # pairing: Node->Neighbors

    def has_edge(self, edge):
        """检查在这张图中是否给定了边"""
        u, v = edge
        return (u, v) in self.edge_properties and (v, u) in self.edge_properties

    def edge_weight(self, edge):
        """返回对应边的权重"""
        return self.get_edge_properties(edge).setdefault(self.weight_attribute_name, self.default_weight)

    def neighbors(self, node):
        # 返回给的节点附近的节点
        return self.node_neighbors[node]

    def has_node(self, node):
        # 检查图里是否给了对应的节点
        return node in self.node_neighbors

    def add_edge(self, edge, word=1, label="", attrs=None):
        # 给这一个图 增加边
        if not attrs:
            attrs = []
        u, v = edge
        if v not in self.node_neighbors[u] and u not in self.node_neighbors[v]:
            self.node_neighbors[u].append(v)
            if u!=v:
                self.node_neighbors[v].append(u)
            self.add_edge_attributes((u, v), attrs)
            self.set_edge_properties((u, v), label=label, weight=word)
        else:
            raise ValueError("Edge (%s, %s) already in graph" % (u, v))

    def add_node(self, node, attrs=None):
        # 给一个图增加节点
        if attrs is None:
            attrs = []
        if node not in self.node_neighbors:
            self.node_neighbors[node] = []
            self.node_attr[node]=attrs
        else:
            raise ValueError("Node %s already in graph" % node)

    def nodes(self):
        """Returns a list of nodes in the graph"""
        return list(self.node_neighbors.keys())

    def edges(self):
        """Returns a list of edges in the graph"""
        return [a for a in list(self.edge_properties.keys())]

    def del_node(self, node):
        """Deletes a given node from the graph"""
        for each in list(self.neighbors(node)):
            if each != node:
                self.del_edge((each, node))
        del(self.node_neighbors[node])
        del(self.node_attr[node])

    def get_edge_properties(self, edge):
        # Helper methods
        return self.edge_properties.setdefault(edge, {})

    def add_edge_attributes(self, edge, attrs):
        """Sets multiple edge attributes"""
        for attr in attrs:
            self.add

    def add_edge_attribute(self, edge, attr):
        """Sets a single edge attribute"""
        self.edge_attr[edge] = self.edge_attributes(edge) + [attr]
        if edge[0] != edge[1]:
            self.edge_attr[(edge[1], edge[0])] = self.edge_attributes((edge[1], edge[0])) + [attr]

    def edge_attributes(self, edge):
        """Returns edge attributes"""
        try:
            return self.edge_attr[edge]
        except KeyError:
            return []

    def del_edge(self, edge):
        """Deletes an edge from the graph"""
        u, v = edge
        self.node_neighbors[u].remove(v)
        self.del_edge_labeling((v, u))

    def del_edge_labeling(self, edge):
        """Deletes the labeling of an edge from the graph"""
        keys = list(edge)
        keys.append(edge[::-1])

        for key in keys:
            for mapping in [self.edge_properties, self.edge_attr]:
                try:
                    del (mapping[key])
                except KeyError:
                    pass