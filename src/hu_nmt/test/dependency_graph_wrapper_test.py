import os
import unittest

from hu_nmt.data_augmentator.dependency_graph_wrapper import DependencyGraphWrapper
from hu_nmt.data_augmentator.utils.data_helpers import get_config_from_yaml

dirname = os.path.dirname(__file__)


class DependencyGraphWrapperTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls) -> None:
        cls.config = get_config_from_yaml(os.path.join(dirname, 'resources/configs/en_test_config.yaml'))
        sentence = 'Helen took the dog for a walk in the woods.'
        cls.dep_graph_wrapper = DependencyGraphWrapper(cls.config, sentence)

    def test_get_root(self):
        actual_root = 'took'
        assert actual_root == self.dep_graph_wrapper.get_root().split('-')[0]  # strip id from node_name

