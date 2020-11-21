package bearmaps.utils.graph;

import bearmaps.utils.graph.streetmap.StreetMapGraph;
import bearmaps.utils.pq.MinHeap;
import bearmaps.utils.pq.MinHeapPQ;
//import org.junit.rules.Stopwatch;
import edu.princeton.cs.algs4.Stopwatch;


import java.util.*;

public class AStarSolver<Vertex> implements ShortestPathsSolver<Vertex>{
    List<Vertex> solution = new ArrayList<Vertex>();
    double timeElapsed; //amount of time that has passed
    double timeLimit; //amount of time we are allotted for solving A* graph
    boolean solved = false; // represents if we were able to solve something
    double totalWeight = 0;
    int numPoll = 0;

    public AStarSolver(AStarGraph<Vertex> input, Vertex start, Vertex end, double timeout) {
        //keep track of elapsedTime
        //use pq to keep track of weights
        //have a list of solutions to update
        //we have access to given edges
        //add vertex start to list of visited edges/vertices
        //everytime we poll
        //if we haven't already looked at neighbor (not in visited), check corresponding edge
        // compare existing heuristic + weight to our current (using estimated... method)
        // if ours is better, update pq with changePriority
        // if not in there we insert
        // add vertex to solution and visited
        // update timeElapsed
            // if timeElapsed is ever greater than timeout, throw exception and say timeElapsed = timeout
            //set total weight equal to 0
        // if we're at the end vertex, change solved to true
        // update pq

        //when we reverse our list in the iteration, add up all the weights in solution to get total weight

        Stopwatch timer = new Stopwatch();
        this.timeLimit = timeout;
        /*timeElapsed = elapsedTime.elapsedTime();
        Collections.reverse(fastestPath);*/
        HashMap<Vertex, WeightedEdge<Vertex>> best = new HashMap<>(); //hold whatever bestDistances has //HashMap<WeightedEdge<Vertex>, Double> best = new HashMap<>();
        HashMap <Vertex, Double> bestDistances = new HashMap<>(); // so that we can easily update the distances
        HashSet<Vertex> visited = new HashSet<>(); //holds just the vertices inside best
        //HashMap<Vertex, Double> distances = new HashMap<>();
        MinHeapPQ<Vertex> pq = new MinHeapPQ<>();
        
        //best will hold vertex to edge
        

        //WeightedEdge<Vertex> edge = new WeightedEdge(null, start, 0);
        Vertex v;
        pq.insert(start, input.estimatedDistanceToGoal(start, end));//edge, input.estimatedDistanceToGoal(start, end));
        bestDistances.put(start, 0.0);
        best.put(start, null);

        while (pq.size() != 0) {
            timeElapsed = timer.elapsedTime();
            if (timeElapsed > timeout) {
                return;
            }

            v = pq.poll();
            numPoll += 1;

            visited.add(v);
                //best.put(edge, distances.get(edge.to()));
                //bestDistances.put(edge.to(), distances.get(edge.to()));
  
            if (v.equals(end)) { //if we've reached the destination
                solved = true;
                //System.out.println("solved!");
                break;
            }

            List<WeightedEdge<Vertex>> neighbors = input.neighbors(v);
            for (WeightedEdge<Vertex> n : neighbors) {
                if (!visited.contains(n.to())) { //only if we haven't visited a vertex already

                    double distance = n.weight();
                    if (bestDistances.containsKey(n.from())) {
                        distance += bestDistances.get(n.from());
                    }
                    /*for (WeightedEdge<Vertex> w : best.keySet()) { //calculate the total distance
                        if (w.to().equals(n.from())) {
                            distance += best.get(w);
                        }
                    }*/

                    //check if we should add or update our distance for this vertex
                    //update the priority queue accordingly
                    if (bestDistances.containsKey(n.to())) {
                        if (distance < bestDistances.get(n.to())) {
                            bestDistances.put(n.to(), distance);
                            best.put(n.to(), n);
                            pq.changePriority(n.to(), distance + input.estimatedDistanceToGoal(n.to(), end));
                        }
                    } else {
                        bestDistances.put(n.to(), distance);
                        best.put(n.to(), n);
                        pq.insert(n.to(), distance + input.estimatedDistanceToGoal(n.to(), end));
                    }

                }
            }
        }

        Vertex curr = end;
        solution.add(curr);
        while (best.get(curr) != null) {
            WeightedEdge<Vertex> edge = best.get(curr);
            curr = edge.from();
            solution.add(curr);
            totalWeight += edge.weight();

            /*for (WeightedEdge<Vertex> path : best.values()) {
                if (path.to().equals(curr)) {
                    solution.add(0, curr);
                    totalWeight += path.weight();
                    if (!solution.contains(start)) {
                       curr = path.from();
                    }
                }
            }*/
        }
        Collections.reverse(solution);


        timeElapsed = timer.elapsedTime(); //idk if this matters for explorationTime
        /*if (Double.compare(timeElapsed, timeout) <= 0) {
            solved = true;
        }*/
    }

    public SolverOutcome outcome() {
        if (timeElapsed > timeLimit) {
            return SolverOutcome.TIMEOUT;
        }
        if (solved) {
            return SolverOutcome.SOLVED;
        }
        return SolverOutcome.UNSOLVABLE;
        /*if (Double.compare(explorationTime(), timeout) == 0 && !solved){
            return SolverOutcome.TIMEOUT;
        }
        if(solved){
            return SolverOutcome.SOLVED;
        }
        return SolverOutcome.UNSOLVABLE;*/
    }

    public List<Vertex> solution() {
        return solution;
    }

    public double solutionWeight() {
        //account for error cases in constructor
        return totalWeight;
    }

    public int numStatesExplored() {
        return numPoll;
    }

    public double explorationTime() {
        return timeElapsed;

    }

    public static void main(String[] args) {
        StreetMapGraph g = new StreetMapGraph("graphTest.in");
    }

}
