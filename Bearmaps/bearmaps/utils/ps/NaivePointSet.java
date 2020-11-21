package bearmaps.utils.ps;

import java.util.List;

public class NaivePointSet implements PointSet {
    List<Point> pointSet;
    public NaivePointSet(List<Point> points) {
        pointSet = points;
    }

    public Point nearest(double x, double y) {
        double minDistance = Double.MAX_VALUE;
        Point given = new Point(x, y);
        Point min = null;
        double dist;

        for (Point point: pointSet) {
            dist = Point.distance(point, given);
            if (dist < minDistance) {
                min = point;
                minDistance = dist;
            }
        }
        return min;
    }

}
