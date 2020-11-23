import unittest
import jumprun
from docopt import docopt

# get docstring for docopt
doc = jumprun.__doc__


class TestJumprun(unittest.TestCase):

    def test_add(self):
        """
        Test the add command
        """
        args = docopt(doc, ['add', 'Test', 'test_file.py'])
        self.assertEqual(args['add'], True)
        self.assertEqual(args['<name>'], 'Test')
        self.assertEqual(args['<command>'], 'test_file.py')

    def test_rm(self):
        """
        Test the remove command
        """
        # test clean entire database
        args = docopt(doc, ['rm', 'all'])
        self.assertEqual(args['rm'], True)

        # test remove single shortcut
        args = docopt(doc, ['rm', 'Test'])
        self.assertEqual(args['rm'], True)
        self.assertEqual(args['<name>'], 'Test')

    def test_show(self):
        """
        Test the show command
        """
        # test show single shortcut
        args = docopt(doc, ['show'])
        self.assertEqual(args['show'], True)

        # test show all shortcuts
        args = docopt(doc, ['show', 'all'])
        self.assertEqual(args['show'], True)

    def test_rename(self):
        """
        Test the rename command
        """
        args = docopt(doc, ['rename', 'Test', 'NewTest'])
        self.assertEqual(args['rename'], True)
        self.assertEqual(args['<old>'], 'Test')
        self.assertEqual(args['<new>'], 'NewTest')

    def test_run(self):
        """
        Test with no main run command
        """
        args = docopt(doc, ['Test'])
        self.assertEqual(args['rename'], False)
        self.assertEqual(args['show'], False)
        self.assertEqual(args['rm'], False)
        self.assertEqual(args['add'], False)

if __name__ == "__main__":
    unittest.main()
