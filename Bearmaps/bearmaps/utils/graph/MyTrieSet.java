package bearmaps.utils.graph;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;

public class MyTrieSet implements TrieSet61BL {

    TrieNode root;
    public class TrieNode {
        boolean end;
        char item;
        HashMap<Character, TrieNode> children;

        public TrieNode(boolean end, char item) {
            this.end = end;
            this.item = item;
            children = new HashMap<Character, TrieNode>();
        }
    }

    public MyTrieSet() {
        root = new TrieNode(false, ' ');
    }

    @Override
    public void clear() {
        root = new TrieNode(false, ' ');
        //root.children = new HashMap<Character, TrieNode>();
    }

    @Override
    public boolean contains(String key) {
        return containsHelper(key, root, 0);
    }

    public boolean containsHelper(String key, TrieNode node, int currentChar) {
        if (node == null) {
            return false;
        }
        if (node.item == key.charAt(currentChar)) {
            if (currentChar == key.length() - 1) {

                return node.end == true;
            }

            return containsHelper(key, node.children.get(key.charAt(currentChar + 1)), currentChar + 1);
        }

        return containsHelper(key,node.children.get(key.charAt(currentChar)), currentChar);
    }

    @Override
    public void add(String key) {
        if (key == null || key.length() < 1) {
            return;
        }
        TrieNode curr = root;
        for (int i = 0, n = key.length(); i < n; i++) {
            char c = key.charAt(i);
            if (!curr.children.containsKey(c)) {
                curr.children.put(c, new TrieNode(false, c));
            }
            curr = curr.children.get(c);
        }
        curr.end = true;
    }

    @Override
    public List<String> keysWithPrefix(String prefix) {
        List<String> keys = new ArrayList<>();
        keysWithPrefixHelper(prefix, root.children.get(prefix.charAt(0)), 0, keys);
        return keys;
    }


    public void keysWithPrefixHelper(String prefix, TrieNode node, int currentChar, List<String> keys) {
        if (node == null) {
            return;
        }
        if (node.item == prefix.charAt(currentChar)) {
            if (currentChar == prefix.length() - 1) {
                traverse(node, keys, prefix);
                return;
            }
            keysWithPrefixHelper(prefix, node.children.get(prefix.charAt(currentChar + 1)), currentChar + 1, keys);
        }
        return;
    }

    public void traverse(TrieNode node, List<String> keys, String prefix) {
        if (node.end) {
            keys.add(prefix);
        }
        for (Character child: node.children.keySet()) {
            traverse(node.children.get(child), keys, prefix + child);
        }
    }

    public String longestPrefixOf(String key) {
        throw new UnsupportedOperationException();
    }

    /*public static void main(String[] args) {
        MyTrieSet t = new MyTrieSet();
        t.add("hello");
        t.add("hi");
        t.add("help");
        t.add("zebra");


        List<String> prefixed = t.keysWithPrefix("b");
        System.out.println(t.contains("he"));
    }*/

}
