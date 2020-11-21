package gitlet;

import java.io.File;
import java.io.IOException;
import java.io.Serializable;
import java.text.ParseException;
import java.text.SimpleDateFormat;
import java.util.*;

/**
 * Driver class for Gitlet, the tiny stupid version-control system.
 *
 * @author
 */
public class Main implements Serializable {
    private static File CWD = new File(".");
//    private static File CWD = new File("./testing/test28-merge-no-conflicts_0");
    private static File hidGit = new File(CWD + "/.gitlet");
    private static File stagingAreaDir = new File(hidGit + "/staging");
    private static File blobs = new File(hidGit + "/blobs");
    private static File commits = new File(hidGit + "/commits");
    private static File gitLog = new File(hidGit + "/gitLog");
    private static File stageFileName = new File(stagingAreaDir + "/stageMapFile");
    private static File removeStage = new File(stagingAreaDir + "/stagedForRemoval");
    private static File headPointer = new File(commits + "/headPointer");

    //    private static Map<String, String> stagedFiles = new HashMap<String, String>();
//    private HashMap <String, Commit> newcommits;
//    private HashMap <String, Commit> commitBranch;
//    private String _head;
//    private String _branch;
//    private ArrayList<String> commitMap;
//    private HashMap<String, String> branchMap;
//
//    private String headBranch;
//    private String headShaCommit;
//    private String directory;
    private static String initialCommitContents = "";
    //    private Commit currentBranch;
    private static File currBranch = new File(hidGit + "/branch");


    private static Map<String, String> stagedFiles = new HashMap<String, String>();
    private static HashMap <String, Commit> newcommits;

//    private static HashMap <String, String> branchMap;
    private static String _head;

    private static String _branch;


    private String headBranch;
    private String headShaCommit;
    private String directory;

    private Commit currentBranch;

//    private static File currenttBranch = new File(hidGit + "/branch");

//    private static File previousBranch = new File(hidGit + "/prevBranch");
    static boolean samebranch = false;

    //checkout should keep track of prevBranch
    //okay so it's kinda complicated but for the first commit that you make with the new branch but like the head pointer is still the old branch
    //so when you do checkout you ONLY update branch like don't touch prevBranch so we can get the sha1 for the parent in the previous branch and once that
    //first commit is over we can just make prevBranch = branch and everything is good
    /**
     * Usage: java gitlet.Main ARGS, where ARGS contains
     * <COMMAND> <OPERAND> ....
     */
    public static void main(String... args) throws IOException, ParseException {
        if (args.length == 0) {
            System.out.println("Please enter a command.");
            return;
        }
        if (!args[0].equals("init") && !hidGit.exists()) {
            System.out.println("Not in an initialized Gitlet directory.");
            return;
        }
        switch(args[0]) {
            case "init":
                init();
                break;
            case "add":
                add(args);
                break;
            case "commit":
                if (args.length < 2) {
                    System.out.println("Please enter a commit message");
                    break;
                }
                commit(args[1]);
                break;
            case "log":
                log();
                break;
            case "find":
                find(args[1]);
                break;
            case "rm":
                rm(args[1]);
                break;
            case "reset":
                reset(args[1]);
                break;
            case "branch":
                branch(args[1]);
                break;
            case "rm-branch":
                rmBranch(args[1]);
                break;
            case "global-log":
                globalLog();
                break;
            case "status":
                status();
                break;
            case "merge":
                if (args.length != 2) {
                    System.out.println("Incorrect Operands.");
                    return;
                }
                merge(args[1]);
                break;
            case "checkout":
                if (args.length > 1) {
                    if (args[1].equals("--")) {
                        checkout(args[2]);
                        break;
                    } else if ((args.length == 4) && (args[2].equals("--"))) {
                        checkout2(args[1], args[3]);
                        break;
                    } else if (args.length == 2) {
                        checkoutBranch(args[1]);
                        break;
                    } else {
                        System.out.println("Incorrect Operands");
                        break;
                    }
                }
        }
    }

    /**
     * Commit saves a snapshot of certain files in the current commit and staging area so they can be restored at a
     * later time, creating a new commit. The commit is said to be tracking the saved files. By default, each commit’s
     * snapshot of files will be exactly the same as its parent commit’s snapshot of files; it will keep versions of
     * files exactly as they are, and not update them.
     *
     * @param message which is the commit message
     * @throws IOException
     */

    public static void commit(String message) throws IOException {
        if (message == null || message.equals("")) {
            System.out.println("Please enter a commit message.");
        }

        if (!stageFileName.exists() && !removeStage.exists()) {
            System.out.println("No changes added to the commit.");
            return;
        }
        String currentBranch = Utils.readContentsAsString(currBranch);

        HashMap<String, String> headPointersMap;
        headPointersMap = Utils.readObject(headPointer, HashMap.class);
        String parentID = headPointersMap.get(currentBranch);

        Commit parentCommit = null;
        HashMap<String, String> commitFiles = null;
        File[] allCommits = commits.listFiles();
        for (File comm : allCommits) {
            if (comm.getName().equals(parentID)) {
                parentCommit = Utils.readObject(comm, Commit.class);
                commitFiles = parentCommit.getCommitFiles();
                break;
            }
        }

        boolean commitChanges = false;
        if (stageFileName.exists()) {
            HashMap<String, String> currStagedFiles = Utils.readObject(stageFileName, HashMap.class);
            if (commitFiles.equals(currStagedFiles)) {
                System.out.println("No changes added to the commit.");
                return;
            }
            Set<String> stagedFileNames = currStagedFiles.keySet();
            for (String addedFile : stagedFileNames) {
                commitFiles.remove(addedFile);
                commitFiles.put(addedFile, currStagedFiles.get(addedFile));
                commitChanges = true;
            }
        }

        if (removeStage.exists()) {
            HashMap<String, String> removeStagedFiles = Utils.readObject(removeStage, HashMap.class);
            Set<String> removedFileNames = removeStagedFiles.keySet();
            for (String removedFile : removedFileNames) {
                if (commitFiles.containsKey(removedFile)) {
                    commitFiles.remove(removedFile);
                    commitChanges = true;
                }
            }
        }

        Commit mostRecent = new Commit(parentID, message, commitFiles, currentBranch);
        String mostRecentSha1 = mostRecent.getCommitSha1();
        addLog(mostRecent, mostRecentSha1, currentBranch);
        headPointersMap.remove(currentBranch);
        headPointersMap.put(currentBranch, mostRecentSha1);
        Utils.writeObject(headPointer, headPointersMap);
        Utils.writeContents(currBranch, currentBranch);

        File mostRecentFile = new File(commits + "/" + mostRecentSha1);
        mostRecentFile.createNewFile();
        Utils.writeObject(mostRecentFile, mostRecent);

        stageFileName.delete();
        removeStage.delete();

        if (!commitChanges) {
            System.out.println("No changes added to the commit.");
        }
    }

    /**
     * addLog is a helper method for log(). It formats and persists a new log entry in the Commit's respective file.
     *
     * @param obj
     * @param sha1
     * @param branch
     * @throws IOException
     */
    private static void addLog(Commit obj, String sha1, String branch) throws IOException {
        List<String> branchLogs = Utils.plainFilenamesIn(gitLog);
        File branchLog = new File(gitLog + "/" + branch);
        if (!branchLogs.contains(branch)) {
            branchLog.createNewFile();
            Utils.writeContents(branchLog, initialCommitContents);
        }
        String prevLog = Utils.readContentsAsString(branchLog);
        String pattern = "EEE MMM dd HH:mm:ss yyyy Z";
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
        TimeZone tz = TimeZone.getTimeZone("America/Los_Angeles");
        simpleDateFormat.setTimeZone(tz);
        String formattedDate = simpleDateFormat.format(obj.getCommitDate());
        String log = "===\ncommit " + sha1 + "\nDate: " + formattedDate + "\n" + obj.getCommitMessage() + "\n\n";
        if (prevLog.equals("")) {
            log = "===\ncommit " + sha1 + "\nDate: " + formattedDate + "\n" + obj.getCommitMessage();
        }
        prevLog = log + prevLog;
        Utils.writeContents(branchLog, prevLog);
    }

    /**
     * logHelper is a helper method to log() since log() cannot take any parameters. However, since we need to
     * print the current branch's log, it is more efficient for us to call this helper method that takes in the
     * branch name as a parameter to print our log.
     *
     * @param branchFile - takes in the branch of the log that we want to print
     */
    private static void logHelper(String branchFile) {
        File f = new File(gitLog + "/" + branchFile);
        String contents = Utils.readContentsAsString(f);
        System.out.println(contents);
    }

    /**
     * Starting at the current head commit, log will display information about each commit backwards along the commit
     * tree until the initial commit, following the first parent commit links, ignoring any second parents
     * found in merge commits.
     */
    public static void log() {
        String currentBranch = Utils.readContentsAsString(currBranch);
        logHelper(currentBranch);
    }

    //iterate through commits directory
    public static void globalLog() {
        //for loop printing commit id, date, and message for all commits
        File[] allCommits = commits.listFiles();
        for (File commitFile : allCommits) {
            if (commitFile.getName().equals("headPointer")) {
                continue;
            }
            Commit curr = Utils.readObject(commitFile, Commit.class);
            printlog(curr);
        }
    }

    /**
     * printLog is a helper method for globalLog. It prints out a log entry of a given commit without the functionality
     * of persisting like addLog has.
     *
     * @param c - takes in a Commit c to produce the log entry from
     */
    public static void printlog(Commit c) {
        String sha1 = c.getCommitSha1();
        String pattern = "EEE MMM dd HH:mm:ss yyyy Z";
        SimpleDateFormat simpleDateFormat = new SimpleDateFormat(pattern);
        TimeZone tz = TimeZone.getTimeZone("America/Los_Angeles");
        simpleDateFormat.setTimeZone(tz);
        String formattedDate = simpleDateFormat.format(c.getCommitDate());
        String log = "===\ncommit " + sha1 + "\nDate: " + formattedDate + "\n" + c.getCommitMessage() + "\n";
        System.out.println(log);
    }

    /**
     * isUnchanged tells you whether a version of a file has been modified from the version that it was in the head
     * commit. It is used as a helper method in add.
     *
     * @param fileName that we want to determine is unchanged or not
     * @return a boolean telling whether or not the file has been modified
     */
    private static boolean isUnchanged(String fileName) {
        HashMap<String, String> headPointers = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        String headSha1 = headPointers.get(currentBranch);

        File[] allCommits = commits.listFiles();
        Commit headCommit = null;

        for (File commit : allCommits) {
            if (commit.getName().equals(headSha1)) {
                headCommit = Utils.readObject(commit, Commit.class);
            }
        }
        HashMap<String, String> commitFiles = headCommit.getCommitFiles();
        if (!commitFiles.containsKey(fileName)) {
            return false;
        }
        String fileSha1 = commitFiles.get(fileName);
        File current = new File(fileName);
        byte[] contents = Utils.readContents(current);


        return fileSha1.equals(Utils.sha1(contents));
    }

    /**
     * restoreFile restores a previously removed file if it has been added back.
     *
     * @param fileName that we want to restore
     */
    private static void restoreFile(String fileName) {
        HashMap<String, String> headPointers = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        String headSha1 = headPointers.get(currentBranch);

        File[] allCommits = commits.listFiles();
        Commit headCommit = null;

        for (File commit : allCommits) {
            if (commit.getName().equals(headSha1)) {
                headCommit = Utils.readObject(commit, Commit.class);
            }
        }
        HashMap<String, String> commitFiles = headCommit.getCommitFiles();
        String fileSha1 = commitFiles.get(fileName);

        File theBlob = new File(blobs + "/" + fileSha1);
        byte[] contents = Utils.readContents(theBlob);

        File f = new File(CWD + "/" + fileName);
        Utils.writeContents(f, contents);
    }

    /**
     * Adds a copy of the file as it currently exists to the staging area (see the description of the commit command).
     *
     * @param files that we want to add
     * @throws IOException
     */
    public static void add(String... files) throws IOException {
        HashMap<String, String> stageMap = new HashMap<String, String>();
        if (stageFileName.exists()) {
            stageMap = Utils.readObject(stageFileName, HashMap.class);
        }
        for (String key : files) {
            if (key.equals("add")) {
                continue;
            }
            if (removeStage.exists()) {
                HashMap<String, String> stagedForRemoval = Utils.readObject(removeStage, HashMap.class);
                if (stagedForRemoval.containsKey(key)) {
                    restoreFile(key);
                    stagedForRemoval.remove(key);
                }
                Utils.writeObject(removeStage, stagedForRemoval);
                return;
            }
            if (isUnchanged(key)) {
                continue;
            }
            File f = new File(key);
            if (!f.exists()) {
                System.out.println("File does not exist");
                return;
            }
            if (stageMap.containsKey(key)) {
                String stagedSha1 = stageMap.get(key);
                String currSha1 = Utils.sha1(Utils.serialize(f));
                if (stagedSha1.equals(currSha1)) {
                    System.out.println("File already exists in staging area");
                    continue;
                }
                makeBlob(f);
                stageMap.remove(key);
                stageMap.put(key, currSha1);
                continue;
            }
            String sha1 = makeBlob(f);
            stageMap.put(key, sha1);
            stageFileName.createNewFile();
            Utils.writeObject(stageFileName, stageMap);
        }
    }

    /**
     * Unstage the file if it is currently staged for addition. If the file is tracked in the current commit,
     * stage it for removal and remove the file from the working directory if the user has not already done so
     * (do not remove it unless it is tracked in the current commit).
     *
     * @param fileName that we want to remove
     * @throws IOException
     */
    public static void rm(String fileName) throws IOException {
        if (!removeStage.exists()) {
            removeStage.createNewFile();
            Utils.writeObject(removeStage, new HashMap<String, String>());
        }

        HashMap<String, String> removalMap = new HashMap<String, String>();
        File removed = new File(CWD + "/" + fileName); //is this right lmao idk
        removalMap = Utils.readObject(removeStage, HashMap.class);

        if (removalMap.containsKey(fileName)) {
            return;
        }

        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        String headSha1 = headPointersMap.get(currentBranch);

        File[] allCommits = commits.listFiles();
        Commit headCommit = null;
        boolean notDeletedFromHead = true;

        /** get the head Commit**/
        for (File commit : allCommits) {
            if (commit.getName().equals(headSha1)) {
                headCommit = Utils.readObject(commit, Commit.class);
                break;
            }
        }

        if (headCommit.getCommitFiles().containsKey(fileName)) {
            String rmSha1 = headCommit.getCommitFiles().get(fileName);
            removalMap.put(fileName, rmSha1);
            notDeletedFromHead = false;
            if (removed.exists()) {
                removed.delete();
            }
        }

        if (!stageFileName.exists() && notDeletedFromHead) {
            System.out.println("No reason to remove the file.");
            return;
        }

        if (!stageFileName.exists()) {
            Utils.writeObject(removeStage, removalMap);
            return;
        }

        HashMap<String, String> currStagedFiles = Utils.readObject(stageFileName, HashMap.class);

        if (currStagedFiles.containsKey(fileName)) {
            currStagedFiles.remove(fileName);
            Utils.writeObject(stageFileName, currStagedFiles);
        }
    }

    /**
     * Prints out the ids of all commits that have the given commit message, one per line. If there are multiple
     * such commits, it prints the ids out on separate lines. The commit message is a single operand;
     * to indicate a multiword message, put the operand in quotation marks.
     *
     * @param commitMessage of the commits we want to find
     */
    public static void find(String commitMessage) {
        if (!commits.exists() || commits.list().length == 0) {
            return;
        }
        File[] commitsListing = commits.listFiles();
        boolean foundCommit = false;
        for (File commit : commitsListing) {
            if (commit.getName().equals("headPointer")) {
                continue;
            }
            Commit curr = Utils.readObject(commit, Commit.class);
            if (commitMessage.equals(curr.getCommitMessage())) {
                System.out.println(curr.getCommitSha1());
                foundCommit = true;
            }
        }
        if (!foundCommit) {
            System.out.println("Found no commit with that message");
        }
    }

    /**
     * status() displays what branches currently exist, and marks the current branch with a *. Also displays what files
     * have been staged for addition or removal.
     */
    public static void status() {
        System.out.println("=== Branches ===");   //get all the keys from the headpointers hashmap in commits dir
        //headPointersMap holds all the branches as keys and their pointer commits as values, so we can just get all the keys
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        Set<String> branches = headPointersMap.keySet();
        String currentBranch = Utils.readContentsAsString(currBranch);
        for (String br : branches) {
            if (br.equals(currentBranch)) {
                System.out.println("*" + br);
                continue;
            }
            System.out.println(br);
        }

        System.out.println("\n=== Staged Files ==="); //go to stagedForAddition file and get all the keys
        if (stageFileName.exists()) {
            HashMap<String, String> addedFiles = Utils.readObject(stageFileName, HashMap.class);
            Set<String> addedFileNames = addedFiles.keySet();
            for (String fileName : addedFileNames) {
                System.out.println(fileName);
            }
        }

        System.out.println("\n=== Removed Files ==="); //go to stagedForRemoval and get all the keys
        if (removeStage.exists()) {
            HashMap<String, String> removedFiles = Utils.readObject(removeStage, HashMap.class);
            Set<String> removedFileNames = removedFiles.keySet();
            for (String fileName : removedFileNames) {
                System.out.println(fileName);
            }
        }

        System.out.println("\n=== Modifications Not Staged For Commit ==="); //extra credit
        System.out.println("\n=== Untracked Files ===");
        System.out.println();
    }

    /**
     * This version of checkout takes the version of the file as it exists in the head commit, the front of the current
     * branch, and puts it in the working directory, overwriting the version of the file that’s already there if there is one.
     * The new version of the file is not staged.
     *
     * @param fileName that we want to checkout
     * @throws IOException
     */
    public static void checkout(String fileName) throws IOException {
        HashMap<String, String> headPointerMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        String headSha1 = headPointerMap.get(currentBranch);

        File[] allCommits = commits.listFiles();
        Commit headCommit = null;

        for (File commit : allCommits) {
            if (commit.getName().equals(headSha1)) {
                headCommit = Utils.readObject(commit, Commit.class);
                break;
            }
        }

        HashMap<String, String> commitFiles = headCommit.getCommitFiles();
        if (!commitFiles.containsKey(fileName)) {
            System.out.println("File does not exist in that commit");
            return;
        }

        String fileSha1 = commitFiles.get(fileName);

        File blob = new File(blobs + "/" + fileSha1);

        File addToCWD = new File(CWD + "/" + fileName);
        if (addToCWD.exists()) {
            addToCWD.delete();
            addToCWD.createNewFile();
        }

        if (blob != null) {
            Utils.writeContents(addToCWD, Utils.readContents(blob));
        }
    }

    /**
     * This version of checkout takes the version of the file as it exists in the commit with the given id,
     * and puts it in the working directory, overwriting the version of the file that’s already there if there is one.
     * The new version of the file is not staged.
     *
     * @param commitID of the version of the file we want to put in the working directory
     * @param fileName file we want to put in the working directory
     * @throws IOException
     */
    public static void checkout2(String commitID, String fileName) throws IOException {
        int commitLength = commitID.length();
        File[] allCommits = commits.listFiles();
        Commit currCommit = null;
        for (File commit : allCommits) {
            if (commit.getName().length() < commitLength) {
                continue;
            }
            String currCommitID = commit.getName().substring(0, commitLength);
            if (currCommitID.equals(commitID)) {
                currCommit = Utils.readObject(commit, Commit.class);
                break;
            }
        }

        if (currCommit == null) {
            System.out.println("No commit with that id exists.");
            return;
        }
        if (currCommit.getCommitFiles() != null && !currCommit.getCommitFiles().containsKey(fileName)) {
            System.out.println("File does not exist in that commit");
            return;
        }

        String fileSha1 = currCommit.getCommitFiles().get(fileName);
        File blob = new File(blobs + "/" + fileSha1);
        File addToCWD = new File(CWD + "/" + fileName);
        Utils.writeContents(addToCWD, Utils.readContents(blob));
    }

    /**
     * This version of checkout takes all files in the commit at the head of the given branch, and puts them in the
     * working directory, overwriting the versions of the files that are already there if they exist. Also, at the end of this command,
     * the given branch will now be considered the current branch (HEAD). Any files that are tracked in the current
     * branch but are not present in the checked-out branch are deleted. The staging area is cleared, unless the
     * checked-out branch is the current branch.
     *
     * @param branchName that we want to checkout
     * @throws IOException
     */
    public static void checkoutBranch(String branchName) throws IOException {
        //first get all the files in the headCommit of the give branchName
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        Set<String> branches = headPointersMap.keySet();
        String currentBranch = Utils.readContentsAsString(currBranch);
        if (!branches.contains(branchName)) {
            System.out.println("No such branch exists.");
            return;
        }
        if (branchName.equals(currentBranch)) {
            System.out.println("No need to checkout the current branch.");
            return;
        }
        //get sha1's of both current branch and given branch name
        String headCommitOfBranchName = headPointersMap.get(branchName);
        String headCommitOfCurr = headPointersMap.get(currentBranch);
        File[] allCommits = commits.listFiles();

        File newBranchCommit = new File(commits + "/" + headCommitOfBranchName);
        File currBranchCommit = new File(commits + "/" + headCommitOfCurr);

        HashMap<String, String> newBranchCommitFiles = Utils.readObject(newBranchCommit, Commit.class).getCommitFiles();
        HashMap<String, String> currBranchCommitFiles = Utils.readObject(currBranchCommit, Commit.class).getCommitFiles();


        Set<String> branchFileNames = newBranchCommitFiles.keySet();
        Set<String> currFileNames = currBranchCommitFiles.keySet();
        HashMap<String, String> currStagedFiles = null;

        for (String branchFile : branchFileNames) {
            String sha1OfTracked = newBranchCommitFiles.get(branchFile);
            if ((!currFileNames.contains(branchFile)) && new File(branchFile).exists()) {  /**please write a new helper method here to help figure if file has been modified (compare sha1 CWD and branchFile**/
                String sha1BranchFile = Utils.sha1(Utils.readContents(new File(branchFile)));
                File blobBranchFile = new File(blobs + "/" + newBranchCommitFiles.get(branchFile));
                String sha1OfBlob = Utils.sha1(Utils.readContents(blobBranchFile));

                if (!(sha1BranchFile.equals(sha1OfBlob))) {
                    System.out.println("There is an untracked file in the way; delete it, or add and commit it first.");
                    return;
                }
            }

            File currBlob = new File(blobs + "/" + sha1OfTracked);
            File addToCWD = new File(CWD + "/" + branchFile);
            addToCWD.createNewFile();
            byte[] contents = Utils.readContents(currBlob);
            Utils.writeContents(addToCWD, contents);
        }

        for (String headCommitFile : currFileNames) {
            if (!branchFileNames.contains(headCommitFile)) {
                File curr = new File(CWD + "/" + headCommitFile);
                curr.delete();
            }
        }

        stageFileName.delete();
        currentBranch = branchName;
        Utils.writeContents(currBranch, currentBranch);
    }

    /**
     * Branch() creates a new branch with the given name, and points it at the current head node. A branch is nothing
     * more than a name for a reference (a SHA-1 identifier) to a commit node. This command does NOT immediately switch
     * to the newly created branch (just as in real Git). Before you ever call branch, your code should be running with
     * a default branch called “master”.
     *
     * @param branchName
     */
    public static void branch(String branchName) {
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        if (headPointersMap.containsKey(branchName)) {
            System.out.println("A branch with that name already exists.");
            return;
        }
        String headForNewBranch = headPointersMap.get(currentBranch);
        headPointersMap.put(branchName, headForNewBranch);
        Utils.writeObject(headPointer, headPointersMap);
    }

    /**
     * This method deletes the branch with the given name. This only means to delete the pointer associated with
     * the branch; it does not mean to delete all commits that were created under the branch, or anything like that.
     *
     * @param branchName
     */
    public static void rmBranch(String branchName) {
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        if (!headPointersMap.containsKey(branchName)) {
            System.out.println("A branch with that name does not exist.");
            return;
        } else if (currentBranch.equals(branchName)) {
            System.out.println("Cannot remove the current branch.");
            return;
        } else {
            headPointersMap.remove(branchName);
            Utils.writeObject(headPointer, headPointersMap);
        }
    }

    /**
     * This method is a helper method to reset, where the user can put in an arbitrary number of characters of the
     * commit ID. It will return the full length version of the shortened version of the commit ID the user
     * may have inputted.
     *
     * @param commitID
     * @return String of the full length version of the commit ID given as it exists as its filename in the commits
     * directory.
     */
    private static String getFullCommitID(String commitID) {
        String fullLengthID = "";
        int commitLength = commitID.length();
        List<String> commitNames = Utils.plainFilenamesIn(commits);
        for (String i : commitNames) {
            if (i.equals("headPointer")) {
                continue;
            }
            String shortened = i.substring(0, commitLength);
            if (shortened.equals(commitID)) {
                return i;
            }
        }
        return null;
    }

    /**
     * This method checks out all the files tracked by the given commit. Removes tracked files that are not present
     * in that commit. Also moves the current branch’s head to that commit node. See the intro for an example of what
     * happens to the head pointer after using reset. The [commit id] may be abbreviated as for checkout.
     * The staging area is cleared. The command is essentially checkout of an arbitrary commit that
     * also changes the current branch head.
     *
     * @param commitID
     * @throws IOException
     */
    public static void reset(String commitID) throws IOException {
        String currentBranch = Utils.readContentsAsString(currBranch);
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranchSha1 = headPointersMap.get(currentBranch);

        String fullLengthID = getFullCommitID(commitID);
        if (fullLengthID == null) {
            System.out.println("No commit with that id exists.");
            return;
        }
        File wantedCommit = new File(commits + "/" + fullLengthID);
        File headCommit = new File(commits + "/" + currentBranchSha1);
        Commit currCommit = Utils.readObject(headCommit, Commit.class);
        Commit wantedCommits = Utils.readObject(wantedCommit, Commit.class);
        HashMap<String, String> currFileNames = currCommit.getCommitFiles();
        HashMap<String, String> wantedCommitFiles = wantedCommits.getCommitFiles();

        for (String commitFile : wantedCommitFiles.keySet()) {
            if ((!currFileNames.containsKey(commitFile)) && new File(commitFile).exists()) {
                String sha1BranchFile = Utils.sha1(Utils.readContents(new File(commitFile)));
                File blobBranchFile = new File(blobs + "/" + wantedCommitFiles.get(commitFile));
                String sha1OfBlob = Utils.sha1(Utils.readContents(blobBranchFile));

                if (!(sha1BranchFile.equals(sha1OfBlob))) {
                    System.out.println("There is an untracked file in the way; delete it, or add and commit it first.");
                    return;
                }
                checkout2(commitID, commitFile);
            }
        }

        for (String headCommitFile : currFileNames.keySet()) {
            if (!wantedCommitFiles.containsKey(headCommitFile)) {
                File curr = new File(CWD + "/" + headCommitFile);
                curr.delete();
            }
        }
        headPointersMap.remove(currentBranch);
        headPointersMap.put(currentBranch, fullLengthID);
        if (stageFileName.exists()) {
            stageFileName.delete();
        }
        Utils.writeObject(headPointer, headPointersMap);
    }

    /**
     * The method creates a blob in the blobs directory in .gitlet. Essentially, it allows us to persist/capture a
     * file's state at a particular point in time. This method is also used as a helper method for add.
     *
     * @param f - File that you want to make a blob of
     * @return the sha1 of the blob that has just been created. This sha1 is then used for other functionalities
     * within the add method.
     * @throws IOException
     */
    private static String makeBlob(File f) throws IOException {
        byte[] contents = Utils.readContents(f);
        File aNewBlob = new File(blobs + "/" + Utils.sha1(contents));
        Utils.writeContents(aNewBlob, contents);
        return Utils.sha1(contents);
    }

    /**
     * Init creates a new Gitlet version-control system in the current directory. This system will automatically
     * start with one commit: a commit that contains no files and has the commit message initial commit
     * (just like that, with no punctuation). It will have a single branch: master, which initially points to this
     * initial commit, and master will be the current branch.
     *
     * @throws IOException
     * @throws ParseException
     */
    public static void init() throws IOException, ParseException {
        if (!hidGit.exists()) {
            hidGit.mkdir();
            commits.mkdir();
            stagingAreaDir.mkdir();
            blobs.mkdir();
            gitLog.mkdir();

            //creates initial commit puts serialized version in the commit folder
            Commit initial = new Commit(null, "initial commit", new HashMap<String, String>(), "master");
            SimpleDateFormat formatter = new SimpleDateFormat("dd-M-yyyy hh:mm:ss a", Locale.ENGLISH);
            formatter.setTimeZone(TimeZone.getTimeZone("UTC"));

            String dateInString = "01-01-1970 00:00:00 AM";
            Date date = formatter.parse(dateInString);

            initial.setCommitDate(date);
            addLog(initial, initial.getCommitSha1(), "master");
            initialCommitContents = Utils.readContentsAsString(new File(gitLog + "/master"));

            File initialCommit = new File(commits + "/" + initial.getCommitSha1());

            initialCommit.createNewFile();
            headPointer.createNewFile();
            currBranch.createNewFile();

            Utils.writeContents(currBranch, "master");
//            Utils.writeContents(previousBranch, "master");
            HashMap<String, String> headPointersMap = new HashMap<String, String>();

            headPointersMap.put("master", initial.getCommitSha1());
            Utils.writeObject(headPointer, headPointersMap);
            Utils.writeObject(initialCommit, initial);
        } else {
            System.out.println("A Gitlet version-control system already exists in the current directory");
        }
    }
    public static void saveCommit(Commit commit, String sha) {
        File f = new File(commits + "/" + sha);
        Utils.writeObject(f, commit);
    }

    //make a commit using the parent1 head commit files and parent2 head commit files

    public static HashMap<String, String> mergeUniqueFiles(String parent2) {
        HashMap<String, String> branchMap = Utils.readObject(headPointer, HashMap.class);
        HashMap<String, String> currCommitFiles = getHeadCommit().getCommitFiles();
        HashMap<String, String> wantedCommitFiles = getCommit(branchMap.get(parent2)).getCommitFiles();
        HashMap<String, String> uniqueFiles =  new HashMap<String, String>();

        for(String fileName: currCommitFiles.keySet()){
            if(!wantedCommitFiles.containsKey(fileName)){
                uniqueFiles.put(fileName, currCommitFiles.get(fileName));
            }
        }

        for(String fileName: wantedCommitFiles.keySet()){
            if(!currCommitFiles.containsKey(fileName)){
                uniqueFiles.put(fileName, wantedCommitFiles.get(fileName));
            }
        }

        return uniqueFiles;

//        Commit commit = new Commit (message, parent1, parent2);
//        if (commit == null) {
//            return;
//        }
//        String sha = Utils.sha1(Utils.serialize(commit));
//        currCommitFiles.add(sha);
//        saveCommit(commit, sha);

        /**
        String branche = getCurrBranch() ;
        branchMap.remove(branche);
        branchMap.put(branche, sha);*/   //DO IN MERGE!!!!!!!!

//        String parent1 = branchMap.get(branchname).substring(0, 7);
//        String msg = "Merged " + branchname + " into " + getCurrBranch() + ".";
//        if (isConflict) {
//            System.out.println("Encountered a merge conflict.");
//        }
    }


    // error messages for merge
    private static boolean isErrorMessage(String branchName) {
        HashMap<String, String> branchMap = Utils.readObject(headPointer, HashMap.class);
        String currentBranch = Utils.readContentsAsString(currBranch);
        HashMap<String, String> currStagedFiles = Utils.readObject(stageFileName, HashMap.class);
        HashMap<String, String> stagedForRemoval = Utils.readObject(removeStage, HashMap.class);
        if (!currStagedFiles.isEmpty() || !stagedForRemoval.isEmpty()) {
            System.out.println("You have uncommitted changes.");
            return true;
        }
        if (!branchMap.containsKey(branchName)) {
            System.out.println("A branch with that name does not exist.");
            return true;
        }
        if (branchName.equals(currentBranch)) {
            System.out.println("Cannot merge a branch with itself.");
            return true;
        }
        Commit current = getHeadCommit();
        Commit given = getCommit(branchMap.get(branchName));
        assert given != null;
        Commit splitPoint = splitPoint(current, given);

        if (given.equals(splitPoint)) {
            System.out.println("Given branch is an ancestor"
                    + " of the current branch.");
            return true;
        }
        if (current.equals(splitPoint)) {
            Utils.writeContents(currBranch, branchName);
            System.out.println("Current branch fast-forwarded.");
            return true;
        }
        return false;
    }

    /** check if there is any file is untracked.
     * @param given the given commit.
     * @return if it is untracked.*/
    public static boolean isUntracked(Commit given, File f) {
        Commit current = getHeadCommit();
        HashMap<String, String> headCommitFiles = current.getCommitFiles();
        HashMap<String, String> givenCommitFiles = given.getCommitFiles();

            if ((!givenCommitFiles.containsKey(f.getName())) && f.exists()) {
                String sha1CommitFile = Utils.sha1(Utils.readContents(f));
                File blobBranchFile = new File(blobs + "/" + givenCommitFiles.get(f.getName()));
                String sha1OfBlob = Utils.sha1(Utils.readContents(blobBranchFile));

                if (!(sha1CommitFile.equals(sha1OfBlob))) {
                    System.out.println("There is an untracked file in the way; delete it, or add and commit it first.");
                    return true;
                }
            }
                return false;
        }



    /*
    gets commit of given sha1
     */
    public static Commit getCommit(String sha) {
        File[] allCommits = commits.listFiles();
        for(File f : allCommits){
            if(f.getName().equals(sha)){
                Commit theOne = Utils.readObject(f, Commit.class);
                return theOne;
            }
        }
        return null;
//        File f = new File(commits + "/" + sha);
//        return Utils.readObject(f, Commit.class);
    }

    // get the head commit.
    public static Commit getHeadCommit() {
        HashMap<String, String> headPointersMap = Utils.readObject(headPointer, HashMap.class);
        return getCommit(headPointersMap.get(getCurrBranch()));
    }

    public static String getHeadBranch() {
        String branch = Utils.readContentsAsString(currBranch);
        return branch;
    }

    public static String getCurrBranch(){
        String currentBranch = Utils.readContentsAsString(currBranch);
        return currentBranch;
    }


    public static void write(String fileName, String givenFileSHA) throws IOException {
        if(!stageFileName.exists()){
            stageFileName.createNewFile();
            Utils.writeObject(stageFileName, new HashMap<String, String>());
        }
        HashMap<String, String> currStagedFiles = Utils.readObject(stageFileName, HashMap.class);
        currStagedFiles.put(fileName, givenFileSHA);
    }

    // write the contents of conflict into working directory.
    // fileName is the file's name.
    // givenFileSHA is the sha of the given file.
    public static String writeConflict1(String fileName, String givenFileSHA) throws IOException {
        File workingFile = new File(CWD + "/" + fileName);
        File givenFile = new File(blobs + "/" + givenFileSHA);
        String givenContents = Utils.readContentsAsString(givenFile);
        Utils.writeContents(workingFile, "<<<<<<< HEAD"
                + System.lineSeparator()
                + "=======" + System.lineSeparator() + givenContents
                + ">>>>>>>" + System.lineSeparator());
        add(fileName);
        byte[] contents = Utils.readContents(workingFile);
        String conflictFileSha1 = Utils.sha1(contents);
        return conflictFileSha1;
    }

    //write the contents of conflict into working directory.
    //fileName is the file's name.
    // givenFileSHA is the sha of the given file.
    // currFileSHA is the sha of the current file
    public static String writeConflict2(String fileName, String currFileSHA, String givenFileSHA) throws IOException {
        File workingFile = new File(CWD + "/" + fileName);
        File currFile = new File(blobs + "/" + currFileSHA);
        File givenFile = new File(blobs + "/" + givenFileSHA);
        String currContents = Utils.readContentsAsString(currFile);
        String givenContents = Utils.readContentsAsString(givenFile);
        Utils.writeContents(workingFile, "<<<<<<< HEAD\n"
                + currContents + "=======\n" + givenContents
                + ">>>>>>>\n");
        byte[] contents = Utils.readContents(workingFile);
        String workingFileSha1 = Utils.sha1(contents);
        add(fileName);
        return workingFileSha1;
    }

    // write the contents of conflict into working directory
    // fileName is the file's name
    // currFileSHA is the sha of the given file
    public static void writeConflict3(String fileName, String currFileSHA) throws IOException {
        File workingFile = new File(CWD + "/" + fileName);
        File currFile = new File(blobs + "/" + currFileSHA);
        String currContents = Utils.readContentsAsString(currFile);
        Utils.writeContents(workingFile, "<<<<<<< HEAD"
                + System.lineSeparator()
                + currContents + "=======" + System.lineSeparator()
                + ">>>>>>>\n");
        add(fileName);
    }

    //givenFileSHA the sha of given branch, splitFileSHA the sha of split point
    // currFileSHA the sha of current branch
    // will return if true
    public static boolean pred1(String givenFileSHA, String splitFileSHA, String currFileSHA) {
        return !givenFileSHA.equals(splitFileSHA)
                && splitFileSHA.equals(currFileSHA);
    }

    //givenFileSHA the sha of given branch, splitFileSHA the sha of split point
    // currFileSHA the sha of current branch
    // will return if true
    public static boolean pred2(String givenFileSHA, String splitFileSHA, String currFileSHA) {
        return !givenFileSHA.equals(splitFileSHA)
                && splitFileSHA.equals(currFileSHA);
    }

    //givenFileSHA the sha of given branch, splitFileSHA the sha of split point
    // currFileSHA the sha of current branch
    // will return if true
    public static boolean pred3(String givenFileSHA, String splitFileSHA, String currFileSHA) {
        return !givenFileSHA.equals(splitFileSHA)
                && givenFileSHA.equals(currFileSHA);
    }

    //givenFileSHA the sha of given branch, splitFileSHA the sha of split point
    // currFileSHA the sha of current branch
    // will return if true
    public static boolean pred4(String givenFileSHA, String splitFileSHA, String currFileSHA) {
        return !givenFileSHA.equals(currFileSHA)
                && !currFileSHA.equals(splitFileSHA)
                && !splitFileSHA.equals(givenFileSHA);
    }

    //givenFileSHA the sha of given branch, splitFileSHA the sha of split point
    // currFileSHA the sha of current branch
    // will return if true
    public static boolean pred5(String givenFileSHA, String splitFileSHA, String currFileSHA) {
        return givenFileSHA == null
                && currFileSHA.equals(splitFileSHA);
    }

    public static Commit splitPoint(Commit current, Commit given) {
        if (given == null) {
            return current;
        }
        int currentBranchLength = 1;
        int givenBranchLength = 1;

        File[] allCommits = commits.listFiles();
        for(File commitFile : allCommits) {
            Commit commit = Utils.readObject(commitFile, Commit.class);
            if(commit.getBranch().equals(current)) {
                currentBranchLength++;
            }
            if(commit.getBranch().equals(givenBranchLength)) {
                givenBranchLength++;
            }
        }

        if(current.getBranch().equals("master")) {
            currentBranchLength--; }
        if(given.getBranch().equals("master")) {
            givenBranchLength--; }

        if(currentBranchLength<=givenBranchLength){
            return current; }
        else{
            return given; }

//        int len = currentBranchLength - givenBranchLength;
//        if (len < 0) {
////            given = given.shorten(-len);
////        } else {
////            current = current.shorten(len);
////        }
//        while (true) {
//            assert current != null;
//            if (!current.equals(given)) break;
//            current = getCommit(current.getParentID());
//            given = getCommit(given.getParentID());
//        }
//        return current;
//    }

//        Set<Commit> seen = new HashSet<>();
//        while (current != null || given != null) {
//            if (current != null) {
//                if (seen.contains(current)) {
//                    return current;
//                }
//                seen.add(current);
//                current = getCommit(current.getParentID());
//            }
//            if (given != null) {
//                if (seen.contains(given)) {
//                    return given;
//                }
//                seen.add(given);
//                given = getCommit(given.getParentID());
//            }
//        }
//        throw new GitletException("Split point not found");
//        return null;
    }

    // does actual merging
    public static void merge(String branchname) throws IOException {
        if(isErrorMessage(branchname)){
            return;
        }
        HashMap<String, String> branchMap = Utils.readObject(headPointer, HashMap.class);
        Commit given = getCommit(branchMap.get(branchname));
        assert given != null;
        Commit splitPoint = splitPoint(getHeadCommit(), given);
        HashMap<String, String> currCommitFiles = getHeadCommit().getCommitFiles();
        HashMap<String, String> givenCommitFiles = given.getCommitFiles();
        HashMap<String, String> splitPointCommitFiles = splitPoint.getCommitFiles();

        HashMap<String, String> mergeCommitFiles = mergeUniqueFiles(branchname);
        HashMap<String, String> conflictFiles = new HashMap<String, String>();
        //do you need to check for untracked files?
        for (String commitFile : currCommitFiles.keySet()) {
            File commit = new File(CWD + "/" + currCommitFiles.get(commitFile));
            if (isUntracked(given, commit)) {
                return;
            }
        }
        //mergeUniqueFiles --> adds unique files of two branches head commits into a HashMap
        boolean isConflict = false;
        for (String fileName : givenCommitFiles.keySet()) {
            String currFileSHA = currCommitFiles.get(fileName);
            String givenFileSHA = givenCommitFiles.get(fileName);
            String splitFileSHA = splitPointCommitFiles.get(fileName);
            if (currFileSHA == null && splitFileSHA == null) {
                write(fileName, givenFileSHA);
                continue;
            }
            if (pred1(givenFileSHA, splitFileSHA, currFileSHA)) {
                String conflictFile1 = writeConflict1(fileName, givenFileSHA);
                mergeCommitFiles.put(fileName, conflictFile1);
                conflictFiles.put(fileName, conflictFile1);
                isConflict = true;
            }
            if (givenFileSHA.equals(splitFileSHA)) {
                mergeCommitFiles.put(fileName, currFileSHA);
                continue;
            }
            if (pred2(givenFileSHA, splitFileSHA, currFileSHA)) {
                write(fileName, givenFileSHA);
                continue;
            }
            if (pred3(givenFileSHA, splitFileSHA, currFileSHA)) {
                continue;
            }
            if (pred4(givenFileSHA, splitFileSHA, currFileSHA)) {
                String mergedFileSha1 = writeConflict2(fileName, currFileSHA, givenFileSHA);
                mergeCommitFiles.put(fileName, mergedFileSha1);
                conflictFiles.put(fileName, mergedFileSha1);
                isConflict = true;
            }
        }
        for (String fileName : currCommitFiles.keySet()) {
            String currFileSHA = currCommitFiles.get(fileName);
            String givenFileSHA = givenCommitFiles.get(fileName);
            String splitFileSHA = splitPointCommitFiles.get(fileName);
            if (pred5(givenFileSHA, splitFileSHA, currFileSHA)) {
                File file = new File(CWD + "/" + fileName);
                if(stageFileName.exists()){
                    HashMap<String, String> currStagedFiles = Utils.readObject(stageFileName, HashMap.class);
                    if(currStagedFiles.containsKey(fileName)){
                        currStagedFiles.remove(fileName);
                    }
                }
                if(removeStage.exists()){
                    HashMap<String, String> stagedForRemoval = Utils.readObject(removeStage, HashMap.class);
                    if(stagedForRemoval.containsKey(fileName)){
                        stagedForRemoval.remove(fileName);
                    }
                }
                file.delete();
            }
        }

        for (String fileName : conflictFiles.keySet()) {
            String currFileSHA = currCommitFiles.get(fileName);
            String givenFileSHA = givenCommitFiles.get(fileName);
            String splitFileSHA = splitPointCommitFiles.get(fileName);
            if (splitFileSHA != null && givenFileSHA == null
                    && !currFileSHA.equals(splitFileSHA)) {
                writeConflict3(fileName, currFileSHA);
                isConflict = true;
            }
        }

        String parent1 = branchMap.get(branchname).substring(0, 7);
        String msg = "Merged " + branchname + " into " + getCurrBranch() + ".";
        if (isConflict) {
            System.out.println("Encountered a merge conflict.");
            return;
        }
        Commit mergeCommit = new Commit(branchMap.get(getCurrBranch()),
                "Merged " + branchname + " into " + getCurrBranch() + ".",
                mergeCommitFiles,
                getCurrBranch());
        String mergeCommitSha1 = mergeCommit.getCommitSha1();

        File mergeCommitFile = new File(commits + "/" + mergeCommitSha1);
        Utils.writeObject(mergeCommitFile, mergeCommit);
        branchMap.remove(getCurrBranch());
        branchMap.put(getCurrBranch(), mergeCommitSha1);
        //make commit
    }
}

