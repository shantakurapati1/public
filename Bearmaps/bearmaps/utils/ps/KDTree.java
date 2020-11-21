package bearmaps.utils.ps;
import java.util.List;
import java.lang.Double;
import java.util.Comparator;

public class KDTree implements PointSet {
    Node root;
    static Comparator<Point> compX = (Point p1, Point p2) -> {
       return Double.compare(p1.getX(), p2.getX());
    };
    static Comparator<Point> compY = (Point p1, Point p2) -> {
        return Double.compare(p1.getY(), p2.getY());
    };

    public class Node {
        Point p;
        Node left;
        Node right;
        boolean comparingOnX;

        public Node(Point p, Node l, Node r, boolean comparison) {
            this.p = p;
            left = l;
            right = r;
            comparingOnX = comparison;
        }
    }

    //You can assume points has at least size 1.
    public KDTree(List<Point> points){
        for (Point p: points) {
             this.insert(p);
        }
    }

    public void insert(Point p) {
        if (root == null) {
            root = new Node(p, null, null, true);
        } else {
            Node pointer = root;
            while (pointer != null && p != null) {
                if (pointer.p.equals(p)) {
                    return;
                }

                if (pointer.comparingOnX) {
                    if (Double.compare(p.getX(), pointer.p.getX()) >= 0) {
                        if (pointer.right == null) {
                            pointer.right = new Node(p, null, null, false);
                        } else {
                            pointer = pointer.right;
                        }
                    } else {
                        if (pointer.left == null) {
                            pointer.left = new Node(p, null, null, false);
                        } else {
                            pointer = pointer.left;
                        }
                    }
                } else {
                    if (Double.compare(p.getY(), pointer.p.getY()) >= 0) {
                        if (pointer.right == null) {
                            pointer.right = new Node(p, null, null, true);
                        } else {
                            pointer = pointer.right;
                        }
                    } else {
                        if (pointer.left == null) {
                            pointer.left = new Node(p, null, null, true);
                        } else {
                            pointer = pointer.left;
                        }
                    }
                }
            }
        }
    }

    // @source: https://docs.google.com/presentation/d/1lsbD88IP3XzrPkWMQ_SfueEgfiUbxdpo-90Xu_mih5U/edit#slide=id.g54b6d82fee_297_0/
    public Point nearestHelper(Point target, Node current, double bestDistance, Point closest) {
        if (current == null) { //should only reach this case if root is null?
            return closest;
        }

        double dist = Point.distance(target, current.p);
        if (dist < bestDistance) {
            bestDistance = dist;
            closest = current.p;
        }

        Node good = null;
        Node bad = null;
        if (current.comparingOnX) {
            if (compX.compare(target, current.p) < 0) {
                good = current.left;
                bad = current.right;
            } else {
                good = current.right;
                bad = current.left;
            }
        } else {
            if (compY.compare(target, current.p) < 0) {
                good = current.left;
                bad = current.right;
            } else {
                good = current.right;
                bad = current.left;
            }
        }


            closest = nearestHelper(target, good, bestDistance, closest);
            bestDistance = Point.distance(closest, target);


         //pruning
            if (current.comparingOnX) {
                if (Math.abs(current.p.getX() - target.getX()) < Math.sqrt(bestDistance)) {
                    closest = nearestHelper(target, bad, bestDistance, closest);
//                    bestDistance = Point.distance(closest, target);
                }
            } else {
                if (Math.abs(current.p.getY() - target.getY()) < Math.sqrt(bestDistance)) {
                    closest = nearestHelper(target, bad, bestDistance, closest);
//                    bestDistance = Point.distance(closest, target);
                }
            }

        return closest;
    }

    public Point nearest(double x, double y) {
        Point target = new Point(x, y);
        return nearestHelper(target, root, Double.MAX_VALUE, root.p);
    }



}
