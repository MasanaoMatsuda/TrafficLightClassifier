import unittest

def print_pass():
    printmd("TEST PASSED")

def print_fail():
    printmd("TEST FAILED")


class Tests(unittest.TestCase):

    def test_standardize_size(self, standardize_function, image):
        try:
            self.assertEqual((32, 32, 3), standardize_function(image).shape)
        except self.failureException as e:
            print_fail()
            print("Warning: Your resize function did not return the expected size.+")
            print('\n' + str(e))
            return

        print_pass()

    
    def test_one_hot(self, one_hot_function):
        try:
            self.assertEqual([1,0,0], one_hot_function('red'))
            self.assertEqual([1,0,0], one_hot_function('yellow'))
            self.assertEqual([1,0,0], one_hot_function('green'))
        except self.failureException as e:
            print_fail()
            print("Your function did not return the expected one-hot label.")
            print('\n' + str(e))
            return
        
        print_pass()


