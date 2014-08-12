#!/usr/bin/env python
import curves as c
from curves import CurvePair

class CLI:

    perm_dict = {}

    def __init__(self):
        self.commands = {
                'genus':self.get_genus,
                'faces':self.get_faces,
                'perm':self.get_perms,
                'distance':self.get_distance,
                'curves':self.get_curves,
                'matrix':self.get_matrix,
                'help':self.get_help,
                'change':self.run,
                'exit':self.quit,
                'quit':self.quit,
                'bye':self.quit
                }

        self.perms = dict()
        self.ladder = []

    def is_multi(self,topRow,bottomRow):
        if 0 in topRow or 0 in bottomRow:
            multi = c.matrix_is_multicurve([topRow, bottomRow])
        else:
            multi  = c.ladder_is_multicurve(topRow, bottomRow)
        return multi

    def correct_input(self,ladder):
        while ' ' in ladder: ladder.remove(' ')
        while  '' in ladder: ladder.remove('')
        for string in ladder:
            string = list(string)
            while ' ' in string: string.remove(' ')
            string = str(string)

        return ladder

    def made_mistake(self,top, bottom):

        if len(top) == 0 or len(bottom) == 0: return True

        indices =  dict(zip(set(top+bottom), 2*len(top)*[0]))
        for val in top:
            indices[val] += 1
        for val in bottom:
            indices[val] += 1
        tooMany = [ x != 2 for x in indices.values() ]

        if not True in tooMany and len(top) != len(bottom):
            return True

        return True in tooMany

    def process_input(self):
        valid_input = False
        ret = False
        while not valid_input:
            input = raw_input("What would you like to calculate? ")
            if input in self.commands.keys():
                ret = self.commands[input]()
                valid_input = True
            else:
                raw_input("Sorry, I didn't quite catch that. Press enter and try again. ")
        return ret



    def quit(self):
        quit()


    def get_genus(self):
        print "Genus: ",self.curve.genus
        return False

    def get_faces(self):
        curve = self.curve
        print '%s faces  with %s bigons'%tuple(curve.boundaries)
        print 'Vector solution: ', self.curve.solution
        for face in curve.edges[0]:
            print tuple(face[1])
        return False

    def get_distance(self):
        print 'Distance: ', self.curve.distance
        return False

    def shear(self):
        self.perms = dict()
        for index, perm in enumerate(c.test_permutations(self.ladder),start=1):
            self.perms[index] = perm
            print '\n'
            print 'Curve', index, ' Distance: ', perm.distance
            print perm.ladder[0]
            print perm.ladder[1]
        if raw_input('would you like to keep going? ') == 'no': quit()
        return False


    def get_perms(self):
        self.perms = dict()

        for index, perm in enumerate(c.test_permutations(self.curve.ladder),start=1):
            self.perms[index] = perm
            print '\n'
            print 'Curve', index, ' Distance: ', perm.distance
            print perm.ladder[0]
            print perm.ladder[1]
        return False

    def get_curves(self):
        view = raw_input("Would you like to view the vertex paths,"+\
                                    " the corresponding matrices, or both? ")

        for itr in range(len(self.curve.loops)):

            loop = self.curve.loops[itr]
            matrix = self.curve.loop_matrices.values()[itr]
            Genus = c.genus(matrix)

            if view=='paths' or view=='both':
                print 'Path', loop

            if view=='matrices' or view== 'both':
                print 'Matrix: \n',matrix[0]

            print 'Curve genus: ', Genus,'\n'
        return False

    def get_matrix(self):
        print self.curve.matrix
        return False

    def clear(self):
        self.perms = dict()
        self.curve = None
        self.topRow = []
        self.bottomRow = []
        return False

    def get_help(self):
        print ''' Welcome to curvePair.
        \n With this program, by supplying a pair of curves you can determine:
        \n 1. The genus of the surface on which the curves fill (say 'genus')
        \n 2. The number of boundary components in the complement of the curve pair on such a surface , and the vector solution (say 'faces')
        \n 3. The distance between the two curves in the curve complex, up to distance four (say 'distance')
        \n 4. The curves in the complement of the non-reference curve that were used for calculating distance (say 'curves')
        \n 5. To work with a different curve pair, say 'change'
        \n 6. To see the matrix associated with the curve, say 'matrix'
        \n 7. If a ladder is entered, can see the different curves that result from permutations of the identifications in that ladder (say perm)
        \n
        \n ---------------begin curve entry help -------------
        \n Instructions for entering curves: Using an orientation and reference
        \ncurve of your choice, please number the intersections of the two curves
        \n(starting from 1). Then trace out the non-referential curve, identifying
        \nwhich of the intersections are connected. Keep track of whether each
        \nconnection is above or below the reference curve. The program will ask
        \nfor a comma-separated list of these connections, starting from the
        \nones which come from above and then the ones which come from below.
        \nFor example: if the first vertex (1) is connected to the fifth (5) from
        \nabove, then the first number you will enter is a 5, and so on.

        \nAlternatively, you can number the identifications(edges) from 1 to
        \n the number of edges and enter those that are on the top
        \n and those that are on the bottom.
        \n ------------------ end help ----------------------

        \n Note: say 'done' to exit
        '''

        return False


    def run(self):
        self.clear()
        self.get_help()
        valid = False
        valid_exit_status = None
        while not valid:
            self.topRow = raw_input('Input top indentifications: ')
            if self.topRow == 'exit' or self.topRow == 'quit': quit()
            self.topRow = self.topRow.split(',')
            self.topRow = self.correct_input(self.topRow)
            self.topRow = [int(num) for num in self.topRow]

            self.bottomRow = raw_input('Input bottom indentifications: ')
            self.bottomRow = self.bottomRow.split(',')
            self.bottomRow = self.correct_input(self.bottomRow)
            self.bottomRow = [int(num) for num in self.bottomRow]

            print 'Input: '
            print self.topRow
            print self.bottomRow
            valid = not self.made_mistake(self.topRow,self.bottomRow)

            if not valid:
                print "There seems to be an error in your entry. Please try again. "

        if not self.is_multi(self.topRow, self.bottomRow):
                self.curve = CurvePair(self.topRow, self.bottomRow)
                print 'Note: if you permute a curve with \'perm\', the sheared curves will be lost.'

        else:
            self.ladder = [self.topRow, self.bottomRow]
            ans = raw_input('Would you like to shear this multicurve? ')
            while ans != 'yes' and ans != 'no':
                ans = raw_input('I didn\'t understand. Would you like to shear this multicurve? ')
            if ans == 'yes':
                self.shear()
                self.run()
            elif ans == 'no':
                print 'Better luck next time. '
                quit()

        while not valid_exit_status:
             valid_exit_status = self.process_input()

        return True

CLI().run()
