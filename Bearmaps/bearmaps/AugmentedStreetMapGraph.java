package bearmaps;

import bearmaps.utils.Constants;
import bearmaps.utils.graph.MyTrieSet;
import bearmaps.utils.graph.WeightedEdge;
import bearmaps.utils.graph.streetmap.Node;
import bearmaps.utils.graph.streetmap.StreetMapGraph;
import bearmaps.utils.ps.KDTree;
import bearmaps.utils.ps.Point;

import java.util.*;

/**
 * An augmented graph that is more powerful that a standard StreetMapGraph.
 * Specifically, it supports the following additional operations:
 *
 *
 * @author Alan Yao, Josh Hug, ________
 */
public class AugmentedStreetMapGraph extends StreetMapGraph {

    HashMap<Point, Node> pointToNode = new HashMap<>();
    KDTree vertices;
    MyTrieSet locations = new MyTrieSet();
    HashMap<String, Node> cleanedNodes = new HashMap<>(); //maps a cleaned string to a node
    HashMap<String, ArrayList<Node>> locationNodes = new HashMap<>();

    public AugmentedStreetMapGraph(String dbPath) {
        super(dbPath);
        // You might find it helpful to uncomment the line below:
        List<Node> nodes = this.getNodes();
        List<Point> points = new ArrayList<>();
        for (Node n : nodes) {
            Point p = new Point(projectToX(n.lon(), n.lat()), projectToY(n.lon(), n.lat()));
            pointToNode.put(p, n);

            List<WeightedEdge<Long>> neighbors = neighbors(n.id());
            if (neighbors != null) { //if a vertex has a neighbor
                points.add(p);
            }
        }
        vertices = new KDTree(points);
        List<Node> allNodes = getAllNodes();
        for (Node n: allNodes) {
            if (n.name() != null) {
                locations.add(cleanString(n.name()));
                cleanedNodes.put(cleanString(n.name()), n);
                if (locationNodes.containsKey(cleanString(n.name()))) {
                    locationNodes.get(cleanString(n.name())).add(n);
                } else {
                    ArrayList<Node> node = new ArrayList<>();
                    node.add(n);
                    locationNodes.put(cleanString(n.name()), node);
                }
            }
        }
    }


    /**
     * For Project Part III
     * Returns the vertex closest to the given longitude and latitude.
     * @param lon The target longitude.
     * @param lat The target latitude.
     * @return The id of the node in the graph closest to the target.
     */
    public long closest(double lon, double lat) {
        double x = projectToX(lon, lat);
        double y = projectToY(lon, lat);

        Point p = vertices.nearest(x, y);
        return pointToNode.get(p).id();
    }

    /**
     * Return the Euclidean x-value for some point, p, in Berkeley. Found by computing the
     * Transverse Mercator projection centered at Berkeley.
     * @param lon The longitude for p.
     * @param lat The latitude for p.
     * @return The flattened, Euclidean x-value for p.
     * @source https://en.wikipedia.org/wiki/Transverse_Mercator_projection
     */
    static double projectToX(double lon, double lat) {
        double dlon = Math.toRadians(lon - ROOT_LON);
        double phi = Math.toRadians(lat);
        double b = Math.sin(dlon) * Math.cos(phi);
        return (K0 / 2) * Math.log((1 + b) / (1 - b));
    }

    /**
     * Return the Euclidean y-value for some point, p, in Berkeley. Found by computing the
     * Transverse Mercator projection centered at Berkeley.
     * @param lon The longitude for p.
     * @param lat The latitude for p.
     * @return The flattened, Euclidean y-value for p.
     * @source https://en.wikipedia.org/wiki/Transverse_Mercator_projection
     */
    static double projectToY(double lon, double lat) {
        double dlon = Math.toRadians(lon - ROOT_LON);
        double phi = Math.toRadians(lat);
        double con = Math.atan(Math.tan(phi) / Math.cos(dlon));
        return K0 * (con - Math.toRadians(ROOT_LAT));
    }


    /**
     * For Project Part IV (extra credit)
     * In linear time, collect all the names of OSM locations that prefix-match the query string.
     * @param prefix Prefix string to be searched for. Could be any case, with our without
     *               punctuation.
     * @return A <code>List</code> of the full names of locations whose cleaned name matches the
     * cleaned <code>prefix</code>.
     */
    public List<String> getLocationsByPrefix(String prefix) {
        /*locations = new MyTrieSet();

        for (WeightedEdge n: pointToNode.values()) {
            if (n.name() != null) {
                locations.add(cleanString(n.name()));
            }
        }*/
        List<String> cleanedNames = locations.keysWithPrefix(prefix);

        //List<Map<String, Object>> matching = getLocations(cleanString(prefix));
        List<String> assignedNames = new LinkedList<>();
        for (String name: cleanedNames) {
            ArrayList<Node> matchingNodes = locationNodes.get(name);
            for (Node n : matchingNodes) {
                assignedNames.add(n.name());
            }
        }


        return assignedNames;
    }

    /**
     * For Project Part IV (extra credit)
     * Collect all locations that match a cleaned <code>locationName</code>, and return
     * information about each node that matches.
     * @param locationName A full name of a location searched for.
     * @return A list of locations whose cleaned name matches the
     * cleaned <code>locationName</code>, and each location is a map of parameters for the Json
     * response as specified: <br>
     * "lat" -> Number, The latitude of the node. <br>
     * "lon" -> Number, The longitude of the node. <br>
     * "name" -> String, The actual name of the node. <br>
     * "id" -> Number, The id of the node. <br>
     */
    public List<Map<String, Object>> getLocations(String locationName) {
        List<Map<String, Object>> matching = new LinkedList<>();

        ArrayList<Node> matchingNodes = locationNodes.get(cleanString(locationName));


        for (Node n: matchingNodes) {
            String name = n.name();
            if (name != null && cleanString(name).equals(cleanString(locationName))) {
                HashMap<String, Object> location = new HashMap<>();
                location.put("lat", n.lat());
                location.put("lon", n.lon());
                location.put("name", n.name());
                location.put("id", n.id());
                matching.add(location);
            }
        }
        return matching;
    }


    /**
     * Useful for Part III. Do not modify.
     * Helper to process strings into their "cleaned" form, ignoring punctuation and capitalization.
     * @param s Input string.
     * @return Cleaned string.
     */
    private static String cleanString(String s) {
        return s.replaceAll("[^a-zA-Z ]", "").toLowerCase();
    }

        
    /**
     * Scale factor at the natural origin, Berkeley. Prefer to use 1 instead of 0.9996 as in UTM.
     * @source https://gis.stackexchange.com/a/7298
     */
    private static final double K0 = 1.0;
    /** Latitude centered on Berkeley. */
    private static final double ROOT_LAT = (Constants.ROOT_ULLAT + Constants.ROOT_LRLAT) / 2;
    /** Longitude centered on Berkeley. */
    private static final double ROOT_LON = (Constants.ROOT_ULLON + Constants.ROOT_LRLON) / 2;

    public static void main(String[] args) {
        AugmentedStreetMapGraph g = new AugmentedStreetMapGraph(Constants.OSM_DB_PATH);
        List<String> main = g.getLocationsByPrefix("e");
        List<Map<String, Object>> topDog = g.getLocations("");
    }
}
