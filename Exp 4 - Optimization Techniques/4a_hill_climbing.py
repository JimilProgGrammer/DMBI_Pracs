import numpy as np

def generate_random_solution(m=3000, n=50):
    """
    Helper function that generates a random solution,
    the solution is a 1 * m vector where m(ij) is a 
    random number ranging from 1 to n.
    
    :param m: Number of cars to distribute
    :param n: Number of auction sites
    :return :
    """
    solution = np.random.randint(n+1, size=(1, m))
    return solution

def evaluate_solution(solution, m=3000, n=50):
    """
    Helper function to evaluate a solution and assign
    a quality measure score to it.
    Quality measure score is calculated as:
        100 * (1 / ((i % n) - solution[0][i]))
    where, 
        -> i in range(0, m)
    
    :param solution: The solution to evaluate
    :param m: Number of cars to distribute
    :param n: Number of auction sites
    :return :
    """
    score = 0
    for i in range(m):
        diff = abs((i % n) - solution[0][i]) + 1
        score = score + (100 * 1/diff)
    return round(score,2)

def mutate_solution(old_solution, m=3000):
    """
    Generates a mutated solution which is in the
    neighbourhood of the old solution; the neighbourhood
    is defined as solution that differs by at most 1
    at any position.
    
    :param old_solution: the solution whose neighbourhood is to be evaluated
    :param m: the number of cars to distribute
    :return :
    """
    idx = np.random.randint(m, size=1)[0]
    if idx%2 == 0:
        old_solution[0][idx] += 1
    else:
        old_solution[0][idx] -= 1
    return old_solution

def run_hill_climb(iterations=1000, m=3000, n=50):
    """
    Run hill climb optimization technique to get a distribution
    of m cars across n auction sites such that cost is minimized.
    
    :param iterations: time limit of running the search
    :param m: number of cars to distribute
    :param n: number of auction sites
    :return :
    """
    current_solution = generate_random_solution(m=3000, n=50)
    current_score = evaluate_solution(current_solution)
    score_has_not_updated_since = 0
    for i in range(iterations):
        neighbour_solution = mutate_solution(current_solution, m)
        neighbour_score = evaluate_solution(neighbour_solution)

        if neighbour_score == current_score:
            current_solution = mutate_solution(neighbour_solution, m)
            current_score = evaluate_solution(current_solution)
            score_has_not_updated_since = 0
            print("[WARN] Iteration " + str(i) + " NEW SOLUTION FROM NEIGHBOURHOOD; Scores were equal")
        elif neighbour_score > current_score:
            print("[INFO] Iteration " + str(i) + ": HILL CLIMB, Better neighbour found.")
            current_solution = neighbour_solution
            current_score = neighbour_score
            score_has_not_updated_since = 0
        else:
            score_has_not_updated_since += 1
            
        if score_has_not_updated_since == 0.2 * iterations:
            print("[INFO] Score has not updated since the last " + str(score_has_not_updated_since) + "iterations. Stopping now.")
            break
        
        print("[RESULT] Iteration " + str(i) + ": Best score so far: " + str(current_score))
    return current_solution, current_score

if __name__ == "__main__":
    best_solution, best_score = run_hill_climb(iterations=1000, m=3000, n=50)
    print("-------------------------------------------------------------------------")
    print("[RESULT] Best Score: " + str(best_score))
    print("[RESULT] Best Solution: " + str(best_solution))
