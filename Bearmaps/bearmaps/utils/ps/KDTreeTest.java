package bearmaps.utils.ps;

import edu.princeton.cs.algs4.Stopwatch;
import org.junit.Test;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;
import java.util.Random;

import java.util.ArrayList;
import java.util.List;

public class KDTreeTest {

    @Test
    //just making sure naive implementation works
    public void testNaive() {
        Point p1 = new Point(1.1, 2.2); // constructs a Point with x = 1.1, y = 2.2
        Point p2 = new Point(3.3, 4.4);
        Point p3 = new Point(-2.9, 4.2);

        NaivePointSet nn = new NaivePointSet(List.of(p1, p2, p3));
        Point ret = nn.nearest(3.0, 4.0); // returns p2
        System.out.println(ret.getX()); // evaluates to 3.3
        System.out.println(ret.getY()); // evaluates to 4.4
    }

    @Test
    public void testNearestBasic() {
        Point p1 = new Point(2, 3);
        Point p2 = new Point(4, 2);
        Point p3 = new Point(4, 2);
        Point p4 = new Point(4, 5);
        Point p5 = new Point(3, 3);
        Point p6 = new Point(1, 5);
        Point p7 = new Point(4, 4);

        List<Point> lst = List.of(p1, p2, p3, p4, p5, p6, p7);
        KDTree kdt = new KDTree(lst);

        Point p = kdt.nearest(3, 3.1);
        assertTrue(p.equals(p5));
        p = kdt.nearest(2.1, 3);
        assertTrue(p.equals(p1));
        p = kdt.nearest(4.1, 5);
        assertTrue(p.equals(p4));

        p = kdt.nearest(0, 7);
        assertTrue(p.equals(p6));
    }

    @Test
    //@source: https://www.youtube.com/watch?v=lp80raQvE5c&feature=youtu.be
    public void testNearestRandom() {
        List<Point> lst = new ArrayList<>();
        Random ran = new Random(210);
        for (int i = 0; i < 100000; i += 1) {
            lst.add(new Point(ran.nextDouble(), ran.nextDouble()));
        }

        KDTree kdt = new KDTree(lst);
        NaivePointSet nps = new NaivePointSet(lst);
        for (int i = 0; i < 1000; i += 1) {
            Point p = new Point(ran.nextDouble(), ran.nextDouble());
            Point kdPoint = kdt.nearest(p.getX(), p.getY());
            Point naivePoint = nps.nearest(p.getX(), p.getY());
            assertTrue(kdPoint.equals(naivePoint));
        }
    }

    @Test
    public void testThreePoints() {
        List<Point> lst = new ArrayList<>();
        Random ran = new Random(40);
        for (int i = 0; i < 3; i += 1) {
            lst.add(new Point(ran.nextDouble(), ran.nextDouble()));
        }

        KDTree kdt = new KDTree(lst);
        NaivePointSet nps = new NaivePointSet(lst);
        for (int i = 0; i < 3; i += 1) {
            Point p = new Point(ran.nextDouble(), ran.nextDouble());
            Point kdPoint = kdt.nearest(p.getX(), p.getY());
            Point naivePoint = nps.nearest(p.getX(), p.getY());
            assertTrue(kdPoint.equals(naivePoint));
        }
    }

    @Test
    public void testTime() {
        List<Point> lst = new ArrayList<>();
        Random ran = new Random(234);
        for (int i = 0; i < 100000; i += 1) {
            lst.add(new Point(ran.nextDouble(), ran.nextDouble()));
        }

        KDTree kdt = new KDTree(lst);
        NaivePointSet nps = new NaivePointSet(lst);

        Stopwatch timer = new Stopwatch();
        for (int i = 0; i < 1000; i += 1) {
            Point p = new Point(ran.nextDouble(), ran.nextDouble());
            kdt.nearest(p.getX(), p.getY());
        }
        double kdTime = timer.elapsedTime();

        Stopwatch timer2 = new Stopwatch();
        for (int i = 0; i < 1000; i += 1) {
            Point p = new Point(ran.nextDouble(), ran.nextDouble());
            nps.nearest(p.getX(), p.getY());
        }
        double naiveTime = timer2.elapsedTime();

        assertTrue(naiveTime / kdTime > 5);


    }


}