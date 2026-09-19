import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import semantic_region_review_terminal as reader


class TerminalReviewTests(unittest.TestCase):
    def test_save_merges_regions_and_rejects_corrupt_existing_notes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.json'
            reader.save_region(path, 'MapA', 'R01', {'description': 'first'})
            reader.save_region(path, 'MapB', 'R01', {'description': 'second'})
            self.assertEqual(len(reader.load_notes(path)['regions']), 2)
            reader.save_region(path, 'MapA', 'R01', {'description': 'revised'})
            data = reader.load_notes(path)['regions']
            self.assertEqual(data['MapB/R01']['description'], 'second')
            self.assertEqual(data['MapA/R01']['description'], 'revised')
            path.write_text('{broken')
            with self.assertRaises(ValueError):
                reader.save_region(path, 'MapA', 'R01', {})
            self.assertEqual(path.read_text(), '{broken')

    def test_quit_and_cancel_do_not_erase_existing_notes(self):
        rows = [{'map_alias': 'A', 'region_alias': 'R01', 'text': 'test'}] * 2
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.json'
            reader.save_region(path, 'A', 'R01', {'description': 'keep'})
            original = path.read_bytes()
            self.assertFalse(reader.review_region(rows, path, 'A', 'R01', ask=lambda _: 'q', show=lambda _: None))
            self.assertEqual(path.read_bytes(), original)
            answers = iter(['', 'new', '', '', '', '', ''])
            self.assertFalse(reader.review_region(rows, path, 'A', 'R01', ask=lambda _: next(answers), show=lambda _: None))
            self.assertEqual(path.read_bytes(), original)

    def test_complete_manual_input_saves_without_method_or_score_fields(self):
        rows = [{'map_alias': 'A', 'region_alias': 'R01', 'text': '\x1b[2JText\nnext'}]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'notes.json'
            answers = iter(['description', 'mostly coherent', 'some exceptions', 'read more', '', 's'])
            shown = []
            self.assertTrue(reader.review_region(rows, path, 'A', 'R01', ask=lambda _: next(answers), show=shown.append))
            result = reader.load_notes(path)['regions']['A/R01']
            self.assertEqual(result['description'], 'description')
            self.assertNotIn('method', result)
            self.assertNotIn('personal_context_score', result)
            self.assertFalse(any('\x1b' in value for value in shown))


if __name__ == '__main__':
    unittest.main()
