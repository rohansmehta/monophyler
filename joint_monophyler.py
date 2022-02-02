import numpy as np
from scipy.special import comb
from collections import Counter
from scipy.linalg import expm
from numpy.linalg import solve
from ete3 import Tree
from discreteMarkovChain import markovChain
import time
import sys
import pandas
import csv


# This function takes an input state and generates all possible output states 
# that result from a single coalescence event.
def generate_onestep_descendents(state):
    onestep_descendents = []
    # Perform all possible single coalescences
    for i in np.nonzero(state)[0]: # These are the "free" lineages
        if(i == 0):
            if(state[0] > 1):
                newstate = list(state)
                newstate[0] = newstate[0] - 1
                newstate = tuple(newstate)
                onestep_descendents.append(newstate)
        else: # these are the intraspecies coalescences
            newstate = list(state)
            newstate[i] = newstate[i] - 1
            newstate[i-1] = newstate[i-1] + 1
            newstate = tuple(newstate)
            onestep_descendents.append(newstate)
    return(tuple(onestep_descendents))            
        
    
# This function generates all possible output states from a set of input states
def generate_output_states(total_states, states_to_loop_through):
    # Loop through all newly generated states
    to_add = []
    for state in states_to_loop_through:
        all_onestep = generate_onestep_descendents(state)
        # If there's only one descendent of this state, we must untuple it
        if(len(all_onestep) > 0):
            if(len(all_onestep) == 1):
                to_add.append(all_onestep[0])
            else:
                to_add.extend(all_onestep)
    # Then we add all of these one-step descendents to the total_states
    # set as well as:
    new_states = [st for st in to_add if st not in total_states]
    if(len(new_states) == 0):
        return
    else:
        total_states.update(new_states) # I hope this handles empty sets!
        generate_output_states(total_states, new_states)
    return

# This function checks to see if state2 can be obtained from an interspecies 
# coalescence in state1
def check_interspecies(state1, state2):
    checkstate = list(state1)
    checkstate[0] = checkstate[0] - 1
    if(checkstate == list(state2)):
        return(True)
    else:
        return(False)
        
# This function checks to see if state2 can be obtained from an 
# intraspecies coalescence in state1, returns the index if so
def check_intraspecies(state1, state2):
    diff = np.subtract(state2, state1)
    # This diff should be 0 until a 1 immediately followed by a -1 and then more
    # zeros
    counted = Counter(diff)
    negones = counted[-1]
    zeros = counted[0]
    ones = counted[1]
    tot = zeros + ones + negones
    if(tot == len(state1) and ones == 1 and negones == 1 and np.where(diff == 1)[0] == np.where(diff == -1)[0] - 1):
        return(np.where(diff == -1)[0])
    else:
        return(-1)

# This function generates the Q matrix (by first generating the transition matrix) for a set of states
def generate_transition_matrix(states):
    numstates = len(states)
    statesize = len(states[0])
    tm = np.zeros((numstates+1, numstates+1)) # including the failure state
    failure_index = numstates # noting that we start at 0 
    tm[failure_index, failure_index] = 0 # This is an absorbing state 
    for i in range(numstates):
        state1 = states[i]
        N = sum([(m+1)*state1[m] for m in range(statesize)])
        for j in range(numstates):
            state2 = states[j]
            if state1 != state2:
                if(check_interspecies(state1, state2)):
                    tm[i, j] = comb(state1[0], 2)
                else:
                    index = check_intraspecies(state1, state2)
                    if(index != -1):
                        index = index[0]
                    if(index > -1):
                        tm[i, j] = state1[index]*comb(index + 1, 2)
        if(state1[0] != sum(state1)): # This excluded case is a success state and cannot fail.
            tm[i, failure_index] = comb(N, 2)-comb(state1[0], 2)-sum([state1[k]*comb(k+1, 2) for k in range(1, statesize)])
        if(state1[0] == sum(state1) and state1[0] == 1):
            tm[i, i] = 1 # This is an absorbing state
        # Now we create the diagonal elements. subtract the rowsums
        tm[i,i] = tm[i,i] - sum(tm[i]) # Turns this into a Q matrix
    return(tm)
               
# This function maps the input state probabilities to their corresponding 
# positions in the output state vector, which is what is used with the transition
# matrix
def map_input_state_probabilities(input_states, input_state_probabilities, output_states_list):
    mapped_input_state_probabilities = [0] * len(output_states_list)
    for oind in range(len(output_states_list)):
        os = output_states_list[oind]
        for iind in range(len(input_states)):
            iis = input_states[iind]
            if(iis == os):
                isp = input_state_probabilities[iind]
                mapped_input_state_probabilities[oind] = isp
    return(mapped_input_state_probabilities)   
      
# This function takes a list of input states (as a list of tuples) and a list of input state probabilities and
# computes the output state distribution through the transition matrix            
# the list of input states DOES NOT INCLUDE the failure probability, which we track separately for simplicity         
def compute_output_state_distribution(input_states, input_state_probabilities, branch_length, failure_prob):
    output_states_set = set(input_states)
    generate_output_states(output_states_set, input_states)
    output_states_list = list(output_states_set)
    mat = generate_transition_matrix(output_states_list)
    mapped_input_state_probabilities = map_input_state_probabilities(input_states, input_state_probabilities, output_states_list)
    mapped_input_state_probabilities.append(failure_prob)
    if(branch_length == 'root'):
        # # First we make this back into a transition matrix I'm not sure this whole block is at all needed
        # mata = np.array(mat)
        # np.fill_diagonal(mata, 0)
        # rowsums = np.sum(mata, 1)
        # for r in np.where(rowsums == 0):
        #     mata[r,r] = 1
        # for row in range(mata.shape[0]):
        #     mata[row] /= np.sum(mata[row])
        pi1 = obtain_steady_state_with_matrix_exponential(mat, mapped_input_state_probabilities)
        output_state_probabilities = pi1
    else:
        output_state_probabilities = np.dot(mapped_input_state_probabilities, expm(mat*branch_length)) # Note that this includes the failure probability at the end
    output_state_probabilities_nofail = output_state_probabilities[0:(len(output_state_probabilities)-1)]
    new_failure_prob = output_state_probabilities[-1]
    #print(output_states_list)
    #print(output_state_probabilities_nofail)
    #print(new_failure_prob)
    #print(sum(input_state_probabilities)+failure_prob, failure_prob, sum(mapped_input_state_probabilities), sum(output_state_probabilities))
    return([output_states_list, output_state_probabilities_nofail, new_failure_prob])
    
# This function takes two sets of input states and their corresponding probabilities and stitches them together.
def stitch_input_states(left_input_states, left_input_state_probabilities, right_input_states, right_input_state_probabilities, left_failure_prob, right_failure_prob):
    # Note that because the labels are species-specific, no non-mixed labels are shared between the daughter nodes
    # And the free mixed labels are each unique. So we can just add the states together
    stitched_states_precombine = []
    stitched_probs_precombine = []
    for lind in range(len(left_input_states)):
        for rind in range(len(right_input_states)):
            # I think the easiest way to do this is to clunkily do all of them and then combine like states
            stitched_state = map(sum, zip(left_input_states[lind], right_input_states[rind]))
            stitched_prob = left_input_state_probabilities[lind] * right_input_state_probabilities[rind]
            stitched_states_precombine.append(stitched_state)
            stitched_probs_precombine.append(stitched_prob)
    # OK so we're just going to have to brute force it
    stitched_states = []
    stitched_probs = []
    for stindex in range(len(stitched_states_precombine)):
        stst = stitched_states_precombine[stindex]
        ststt = tuple(stst)
        stprob = stitched_probs_precombine[stindex]
        if(ststt not in stitched_states):
            stitched_states.append(ststt)
            stitched_probs.append(stprob)
        else:
            match_index = [x for x in range(len(stitched_states)) if stitched_states[x] == ststt][0]
            stitched_probs[match_index] = stitched_probs[match_index] + stprob
    failure_prob = left_failure_prob*(1-right_failure_prob) + right_failure_prob*(1-left_failure_prob) + right_failure_prob*left_failure_prob
    return([stitched_states, stitched_probs, failure_prob])
    
# This function adds sample sizes to a species tree structure
def add_samples(t, samplenames, samples):
    for node in t.traverse('postorder'):
        if node.name in samplenames:
            node.add_features(samples = [samples[x] for x in range(len(samples)) if samplenames[x] == node.name])

def is_steady_state(state, Q):
    """
    Returns a boolean as to whether a given state is a steady 
    state of the Markov chain corresponding to the matrix Q
    """
    return np.allclose((state @ Q), 0)

def obtain_steady_state_with_matrix_exponential(Q, start, max_t=100):
    """
    Solve the defining differential equation until it converges.
    
    - Q: the transition matrix
    - max_t: the maximum time for which the differential equation is solved at each attempt.
    """
    
    state = start
    
    while not is_steady_state(state=state, Q=Q):
        state = state @ expm(Q * max_t)
    
    return state

# This function performs the recursion
def get_node_output(node, maxsamples):    
    if node.is_leaf():
        input_state = [0]*max(samples)
        input_state[node.samples[0]-1] = 1
        input_state = tuple(input_state)
        input_state_list = [input_state]
        input_state_probabilities = [1]
        branch_length = node.dist
        input_failure_prob = 0
        output = compute_output_state_distribution(input_state_list, input_state_probabilities, branch_length, input_failure_prob)
        return(output)
    else:
        left_input_structure = get_node_output(node.children[0], maxsamples)
        right_input_structure = get_node_output(node.children[1], maxsamples)
        left_input_states = left_input_structure[0]
        right_input_states = right_input_structure[0]
        left_input_probs = left_input_structure[1]
        right_input_probs = right_input_structure[1]
        left_failure_prob = left_input_structure[2]
        right_failure_prob = right_input_structure[2]
        inputs = stitch_input_states(left_input_states, left_input_probs, right_input_states, right_input_probs, left_failure_prob, right_failure_prob)
        input_state_list = inputs[0]
        input_state_probabilities = inputs[1]
        input_failure_prob = inputs[2]
        if node.is_root():
            branch_length = 'root' # note that this just runs until convergence. there should be a better way
        else:
            branch_length = node.dist
        output = compute_output_state_distribution(input_state_list, input_state_probabilities, branch_length, input_failure_prob)
        return(output)
        
inputstring = sys.argv[1]
if(inputstring == '-m' ):
    t = Tree(sys.argv[2])
    tsamplenames = sys.argv[3]
    tsamples = sys.argv[4]          
    #t = Tree("(A:0.0019,(B:0.0004,C:0.00035):0.0016);") # example to test
    samplenames = tsamplenames.split(',')
    csamples = tsamples.split(',')
    samples = [int(s) for s in csamples]
    maxsamples = max(samples)
    add_samples(t, samplenames, samples)
    finalout = get_node_output(t, maxsamples)
    print(1-finalout[2])
elif(inputstring == '-f'):
    treefilename = sys.argv[2]
    samplenamefilename = sys.argv[3]
    samplefilename = sys.argv[4] 
    outputfilename = sys.argv[5]
    treelist = []
    samplenamelist = []
    samplelist = []
    with open(treefilename, newline='') as f1:
        reader = csv.reader(f1, delimiter = ' ')
        for row in reader:
            treelist.append(row)
    with open(samplenamefilename, newline='') as f2:
        reader = csv.reader(f2, delimiter = ' ')
        for row in reader:
            samplenamelist.append(row)
    with open(samplefilename, newline='') as f3:
        reader = csv.reader(f3, delimiter = ' ')
        for row in reader:
            samplelist.append(row)
    f1.close()
    f2.close()
    f3.close()
    with open(outputfilename,'w',  newline='') as f4:
        writer = csv.writer(f4)
        for i in range(len(treelist)):
            tree = Tree(treelist[i][0])
            samplenames = samplenamelist[i][0].split(',')
            csamples = samplelist[i][0].split(',')
            samples = [int(s) for s in csamples]
            maxsamples = max(samples)
            add_samples(tree, samplenames, samples)
            finalout = get_node_output(tree, maxsamples)
            result = 1-finalout[2]
            writer.writerow([result])
else:
    print('Incorrect input. Looking for -m or -f.')

