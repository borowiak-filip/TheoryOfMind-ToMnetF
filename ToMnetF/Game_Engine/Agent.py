import numpy as np
import Env as Env
import heapq
import re
import os
"""
Code taken from https://github.com/Nik-Kras/ToMnet-N
@Author Nikita Krasnytskyi
@Modified by Filip Borowiak
"""
adjacent_squares = ((0, -1), (0, 1), (-1, 0), (1, 0),)

class Node:
    """
    A node class for A* Pathfinding
    """

    def __init__(self, parent=None, position=None):
        self.parent = parent
        self.position = position

        self.g = 0
        self.h = 0
        self.f = 0

    def __eq__(self, other):
        return self.position == other.position

    def __repr__(self):
        return f"{self.position} - g: {self.g} h: {self.h} f: {self.f}"

    # defining less than for purposes of heap queue
    def __lt__(self, other):
        return self.f < other.f

    # defining greater than for purposes of heap queue
    def __gt__(self, other):
        return self.f > other.f

class AgentStar:
    def __init__(self, env: Env, sight, observability="partial", consume_goals=1):
        self.env = env
        self.consume_goals = consume_goals
        self.observability = observability
        if self.observability == "partial": self.sight = min(sight, self.env.height)
        else: self.sight = None
        self.position = env.players_position
        self.memory = np.full((env.width, env.height), None)
        self.trajectory = []
        self.position_trajectory = []
        self.step_picked_goal = []
        self.picked_list = []
        self.goal_found = None
        self.max_goal = self.get_highest_goal(self.picked_list)
        self.picked_goal = False

    """
    PS: How to read Actions:
    0 - UP
    1 - RIGHT
    2 - DOWN
    3 - LEFT
    """

    def get_highest_goal(self, ignore_list):
        max_value = 0
        result_point = None
        for key, value in self.env.GoalValue.items():
            if max_value < value and key not in ignore_list:
                max_value = value
                result_point = key

        return result_point

    def get_highest_goal_from_memory(self, ignore_list):
        value = 0
        result_point = None
        for row in self.memory:
            for point in row:
                if point in self.env.GoalValue and value < self.env.GoalValue[point] and point not in ignore_list:
                    value = self.env.GoalValue[point]
                    result_point = point

        return result_point

    def chose_action(self, observability="partial"):
        if observability == "full" or self.picked_goal:
            self.goal_found = self.get_highest_goal_from_memory([])

        result = self.astar(observability=observability)

        ignore_list = []

        if self.goal_found is None:
            while result == -1:
                self.goal_found = self.get_highest_goal_from_memory(ignore_list)
                """ we need this to solve this problem
                # - #
                # O #
                D A #
                """
                if self.goal_found is None:
                    self.goal_found = self.env.objectsEnum["Wall"]

                ignore_list.append(self.goal_found)
                result = self.astar(observability=observability)

        if self.env.goal_picked != 0:
            self.step_picked_goal.append(len(self.trajectory)-1)
        self.position_trajectory.append(self.position)
        self.trajectory.append(result)
        return result

    def on_pickup(self, reward):
        self.memory[self.env.players_position[0], self.env.players_position[1]] = self.env.objectsEnum["Path"]
        self.picked_goal = True

    # Astar-like algorithm
    def astar(self, observability="partial"):
        """Returns a list of tuples as a path from the given start to the given end in the given maze"""

        # Create start and end node
        start_node = Node(None, position=tuple(self.position))
        start_node.g = start_node.h = start_node.f = 0

        if observability == "full" and not (self.goal_found is None):
            goal_row = goal_col = 0
            for i in range(self.env.height):
                for j in range(self.env.width):
                    if self.memory[i, j] == self.goal_found:
                        goal_row = i
                        goal_col = j
            goal_position = [goal_row, goal_col]
            end_node = Node(None, position=tuple(goal_position))
        # Initialize both open and closed list
        open_list = []
        closed_list = []

        # Heapify the open_list and Add the start node
        heapq.heapify(open_list)
        heapq.heappush(open_list, start_node)

        # Loop until you find the end
        while len(open_list) > 0:
            # Get the current node
            current_node = heapq.heappop(open_list)
            if not (current_node in closed_list):
                closed_list.append(current_node)

            point_obj = self.memory[current_node.position[0], current_node.position[1]]
            # Found the unexplored cell or it's goal
            if point_obj == self.goal_found or point_obj == self.max_goal:
                path = []
                current = current_node
                while current is not None:
                    path.append(current.position)
                    current = current.parent

                try:
                    first_node = path[-2]
                except:
                    return
                if first_node[1] < self.position[1]:
                    return 3
                elif first_node[1] > self.position[1]:
                    return 1
                elif first_node[0] < self.position[0]:
                    return 0
                elif first_node[0] > self.position[0]:
                    return 2

                return -1

            # Generate children
            children = []
            for new_position in adjacent_squares:  # Adjacent squares

                # Get node position
                node_position = (current_node.position[0] + new_position[0], current_node.position[1] + new_position[1])

                # Make sure within range
                if node_position[0] > (self.env.width - 1) or node_position[0] < 0 \
                        or node_position[1] > (self.env.height - 1) or node_position[1] < 0:
                    continue

                # Make sure we don't wall into walls
                point_obj = self.memory[node_position[0], node_position[1]]
                if not (point_obj == self.env.objectsEnum["Path"] or point_obj == self.goal_found or point_obj == self.max_goal):
                    continue

                new_node = Node(current_node, node_position)

                # Append
                children.append(new_node)

            # Loop through children
            for child in children:
                # Child is on the closed list
                if len([closed_child for closed_child in closed_list if closed_child == child]) > 0:
                    continue

                # Create the f, g, and h values
                child.g = current_node.g + 1

                # Use ASTAR for path optimization
                # For Full-observability
                if observability == "full":
                    child.h = ((child.position[0] - end_node.position[0]) ** 2) + ((child.position[1] - end_node.position[1]) ** 2)
                else:
                    child.h = 0
                child.f = child.g + child.h

                # Child is already in the open list
                if len([open_node for open_node in open_list if
                        child.position == open_node.position and child.g > open_node.g]) > 0:
                    continue

                # Add the child to the open list
                heapq.heappush(open_list, child)

        return -1


    def update_world_observation(self):
        position, sight_array = self.env.get_sight(self.sight, self.observability)
        self.position = position

        if self.observability == "full":
            self.memory = sight_array
        else:

            half_sight = int(self.sight * 0.5)
            for i in range(self.sight):
                for j in range(self.sight):
                    x = i + position[0] - half_sight
                    y = j + position[1] - half_sight

                    sight_elem = sight_array[i, j]
                    if -1 < x < self.env.width and -1 < y < self.env.height and sight_elem is not None:
                        self.memory[x, y] = sight_elem

    def render(self):
        # 2. Draw a Map
        graph = ""
        for row in range(self.env.width):
            row_string = ""
            for col in range(self.env.height):

                # Draw player
                if self.position == [row, col]: row_string += u" \u25CB " # u" \u25CC "

                # Draw walls, paths and goals
                else:
                    if   self.memory[row, col] is None:                      row_string += ' ? '  # Unexplored
                    elif self.memory[row, col] == self.env.objectsEnum["Wall"]:   row_string += ' # '  # Wall
                    elif self.memory[row, col] == self.env.objectsEnum["Path"]:   row_string += ' - '  # Path
                    elif self.memory[row, col] == self.env.objectsEnum["Goal A"]: row_string += ' A '  # Goal 1
                    elif self.memory[row, col] == self.env.objectsEnum["Goal B"]: row_string += ' B '  # Goal 2
                    elif self.memory[row, col] == self.env.objectsEnum["Goal C"]: row_string += ' C '  # Goal 3
                    elif self.memory[row, col] == self.env.objectsEnum["Goal D"]: row_string += ' D '  # Goal 4
                    else: print("ERROR: Incorrect map value! Position: " ,row, ", ", col)

            row_string += '\n'
            graph += row_string
        print(graph)
        return graph

    """
        A function saves a game: Maze (walls, goals, player), 
        Consumed Goal, Length of trajectory and each trajectory step to .txt file
    """
    def save_game(self, name="TestFolder"):

        # Shouldn't be here, but it fixes the save of last goal :(
        if self.env.goal_picked != 0:
            self.step_picked_goal.append(len(self.trajectory)-1)
        
        # Get the path to folder
        gf = os.path.join('TomNetF','data', 'Saved Games', name) # path to games folder
        files = os.listdir(gf)
        r = re.compile(".*.txt")
        files = list(filter(r.match, files))

        # Chose the number for the new name
        #print("There are files in the folder: ", files)
        max_number = 0
        for file in files:
            if max_number < int(file[4:-4]):
                max_number = int(file[4:-4])

        #print("The max current number is: ", max_number)
        new_name_number = max_number+1

        # Save the Game line by line
        new_file_path = os.path.join(gf, 'test' + str(new_name_number) + '.txt')
        realmap = self.render()

        with open(new_file_path, 'w') as f:
            f.write('Maze:\n')

            # Save the Maze (Walls, Goals, Player)
            wall_line = '#' * (self.env.height+2)
            f.write(wall_line + '\n')
            for i in range(self.env.width):
                f.write(self.env.init_map[i] + '\n')
            f.write(wall_line + '\n')
            for i, goal in enumerate(self.env.consumed_goal):
                f.write('Goal Consumed #' + str(i+1) + ' : '+ goal +'\n')
            f.write('Trajectory length: ' + str(len(self.trajectory)) + '\n')

            # Save moves
            for i in range(len(self.trajectory)):
                msg = str(self.position_trajectory[i]) + ' : ' + str(self.trajectory[i]) + ' : '
                picked_bool = False
                for j in range(len(self.step_picked_goal)):
                    if self.step_picked_goal[j] == i:
                        msg = msg + self.env.consumed_goal[j]  # If consumed First goal here - mention which
                        picked_bool = True
                if not picked_bool:
                    msg = msg + "X"                         # If didn't consume - put X
                f.write(msg + '\n')

            f.close()

