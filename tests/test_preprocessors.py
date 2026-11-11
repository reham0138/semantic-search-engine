import unittest

from app.core.config import SearchConfig
from app.data.preprocessors import normalize_value, prepare_document


class NormalizeValueTests(unittest.TestCase):

    def test_none_becomes_empty(self):
        self.assertEqual(normalize_value(None), "")

    def test_collapses_whitespace_and_stringifies(self):
        self.assertEqual(normalize_value("  Cheese \n Pizza "), "Cheese Pizza")
        self.assertEqual(normalize_value(3333), "3333")

    def test_joins_list_items(self):
        self.assertEqual(normalize_value(["spicy", None, " ", "hot"]), "spicy hot")


class PrepareDocumentTests(unittest.TestCase):

    def test_keeps_non_empty_fields_in_config_order(self):
        record = {"id": 7, "category": "Pizza", "name": "Cheese Pizza"}

        document = prepare_document(record, SearchConfig())

        self.assertEqual(document["id"], "7")
        self.assertEqual(document["fields"], {"name": "Cheese Pizza", "category": "Pizza"})
        self.assertEqual(document["text"], "Cheese Pizza Pizza")

    def test_missing_id_raises(self):
        with self.assertRaises(ValueError):
            prepare_document({"name": "No id"}, SearchConfig())


if __name__ == "__main__":
    unittest.main()
