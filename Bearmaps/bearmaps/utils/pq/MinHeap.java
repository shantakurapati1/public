package bearmaps.utils.pq;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.HashMap;
import java.util.NoSuchElementException;

/* A MinHeap class of Comparable elements backed by an ArrayList. */
public class MinHeap<E extends Comparable<E>> {

    /* An ArrayList that stores the elements in this MinHeap. */
    private ArrayList<E> contents;
//    private HashMap<Integer, E> contents;
    private HashMap<E, Integer> backwards;
    private int size;
    // TODO: YOUR CODE HERE (no code should be needed here if not 
    // implementing the more optimized version)

    /* Initializes an empty MinHeap. */
    public MinHeap() {
//        contents = new HashMap<>();
        contents= new ArrayList<>();
        backwards = new HashMap<>();
        contents = new ArrayList<>();
        contents.add(null);

//        contents.put(0, null);
        backwards.put(null, 0);
//        setContents = new HashSet<>();
    }

    /* Returns the element at index INDEX, and null if it is out of bounds. */
    private E getElement(int index) {
        if (index >= contents.size()) {
            return null;
        } else {
            return contents.get(index);
        }
    }

    /* Sets the element at index INDEX to ELEMENT. If the ArrayList is not big
       enough, add elements until it is the right size. */
    private void setElement(int index, E element) {
//        if (getElement(index) != null) {
//            setContents.remove(getElement(index));
//        }
        while (index >= contents.size()) {
            contents.add(null);
        }
        contents.set(index, element);
//        contents.put(index, element);
        backwards.put(element, index);

//        if (element != null) {
//            setContents.add(getElement(index));
//        }
    }

    /* Swaps the elements at the two indices. */
    private void swap(int index1, int index2) {
        E element1 = getElement(index1);
        E element2 = getElement(index2);
        setElement(index2, element1);
        setElement(index1, element2);
    }

    /* Prints out the underlying heap sideways. Use for debugging. */
    @Override
    public String toString() {
        return toStringHelper(1, "");
    }

    /* Recursive helper method for toString. */
    private String toStringHelper(int index, String soFar) {
        if (getElement(index) == null) {
            return "";
        } else {
            String toReturn = "";
            int rightChild = getRightOf(index);
            toReturn += toStringHelper(rightChild, "        " + soFar);
            if (getElement(rightChild) != null) {
                toReturn += soFar + "    /";
            }
            toReturn += "\n" + soFar + getElement(index) + "\n";
            int leftChild = getLeftOf(index);
            if (getElement(leftChild) != null) {
                toReturn += soFar + "    \\";
            }
            toReturn += toStringHelper(leftChild, "        " + soFar);
            return toReturn;
        }
    }

    /* Returns the index of the left child of the element at index INDEX. */
    private int getLeftOf(int index) {
        // TODO: YOUR CODE HERE
        return index * 2;
    }

    /* Returns the index of the right child of the element at index INDEX. */
    private int getRightOf(int index) {
        // TODO: YOUR CODE HERE
        return index * 2 + 1;
    }

    /* Returns the index of the parent of the element at index INDEX. */
    private int getParentOf(int index) {
        // TODO: YOUR CODE HERE
        return index / 2;
    }

    /* Returns the index of the smaller element. At least one index has a
       non-null element. If the elements are equal, return either index. */
    private int min(int index1, int index2) {
        // TODO: YOUR CODE HERE
        E first = getElement(index1);
        E second = getElement(index2);
        if (first == null) { //if null then just return the other one?
            return index2;
        }
        if (second == null) {
            return index1;
        }
        if (first.compareTo(second) > 0) {
            return index2;
        } else {
            return index1;
        }
    }

    /* Returns but does not remove the smallest element in the MinHeap. */
    public E findMin() {
        // TODO: YOUR CODE HERE
        return getElement(1);
    }

    /* Bubbles up the element currently at index INDEX. */
    private void bubbleUp(int index) {
        // TODO: YOUR CODE HERE
        E elem = getElement(index);
        E parent = getElement(getParentOf(index));
        int curr = index;
        while (parent != null && elem.compareTo(parent) < 0) {
            swap(curr, getParentOf(curr));
            curr = getParentOf(curr);
            if (curr == 1) {
                return;
            }
            parent = getElement(getParentOf(curr));
        }
    }

    /* Bubbles down the element currently at index INDEX. */
    private void bubbleDown(int index) {
        // TODO: YOUR CODE HERE
        E elem = getElement(index);
        E leftChild;// = getElement(getLeftOf(index));
        E rightChild;// = getElement(getRightOf(index));
        int curr = index;
        boolean swapping = true;
        while (swapping) {
            leftChild = getElement(getLeftOf(curr));
            rightChild = getElement(getRightOf(curr));
            if (rightChild == null && leftChild == null) {
                swapping = false;
            }
            else if (rightChild == null && leftChild != null) {
                if (elem.compareTo(leftChild) > 0) {
                    swap(curr, getLeftOf(curr));
                    curr = getLeftOf(curr);
                } else {
                    swapping = false;
                }
            }
            else if (rightChild != null && leftChild == null) {
                if (elem.compareTo(rightChild) > 0) {
                    swap(curr, getRightOf(curr));
                    curr = getRightOf(curr);
                } else {
                    swapping = false;
                }
            }
            else {
                int compareRight = elem.compareTo(rightChild);
                int compareLeft = elem.compareTo(leftChild);
                if (compareRight > 0 && compareLeft > 0) {
                    if (rightChild.compareTo(leftChild) < 0 ) {
                        swap(curr, getRightOf(curr));
                        curr = getRightOf(curr);
                    } else {
                        swap(curr, getLeftOf(curr));
                        curr = getLeftOf(curr);
                    }
                } else if (compareRight > 0) {
                    swap(curr, getRightOf(curr));
                    curr = getRightOf(curr);
                } else if (compareLeft > 0) {
                    swap(curr, getLeftOf(curr));
                    curr = getLeftOf(curr);
                } else {
                    swapping = false;
                }

            }

        }
    }

    /* Returns the number of elements in the MinHeap. */
    public int size() {
        // TODO: YOUR CODE HERE
        return size;
    }

    /* Inserts ELEMENT into the MinHeap. If ELEMENT is already in the MinHeap,
       throw an IllegalArgumentException.*/
    public void insert(E element) {
        // TODO: YOUR CODE HERE
        if (contains(element)) {
            throw new IllegalArgumentException();
        } else {
            setElement(size() + 1, element);
            size += 1;
            bubbleUp(size());
        }
    }

    /* Returns and removes the smallest element in the MinHeap. */
    public E removeMin() {
        // TODO: YOUR CODE HERE
        E min = getElement(1);
        swap(1, size());
        setElement(size(), null);
        bubbleDown(1);
        size -= 1;
        return min;
    }

    /* Replaces and updates the position of ELEMENT inside the MinHeap, which
       may have been mutated since the initial insert. If a copy of ELEMENT does
       not exist in the MinHeap, throw a NoSuchElementException. Item equality
       should be checked using .equals(), not ==. */
    public void update(E element) {
        // TODO: YOUR CODE HERE
        if (!contains(element)) {
            throw new NoSuchElementException();
        }
        int indexOfElement = backwards.get(element);
        if(getElement(indexOfElement).equals(element)){
            setElement(indexOfElement, element);
            if (getElement(getParentOf(indexOfElement)) != null &&
                    getElement(indexOfElement).compareTo(getElement(getParentOf(indexOfElement))) < 0) { //if current element is greater than its parent
                bubbleUp(indexOfElement);
            }
            if (getElement(getLeftOf(indexOfElement)) != null &&
                    getElement(indexOfElement).compareTo(getElement(getLeftOf(indexOfElement))) > 0) {
                bubbleDown(indexOfElement);
            }
            if (getElement(getRightOf(indexOfElement)) != null &&
                    getElement(indexOfElement).compareTo(getElement(getRightOf(indexOfElement))) > 0) {
                bubbleDown(indexOfElement);
            }
        }
    }

    /* Returns true if ELEMENT is contained in the MinHeap. Item equality should
       be checked using .equals(), not ==. */
    public boolean contains(E element) {
        // TODO: YOUR CODE HERE
        return backwards.containsKey(element);
    }
}
