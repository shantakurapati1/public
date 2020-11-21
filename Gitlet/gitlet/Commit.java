package gitlet;

import java.io.File;
import java.io.Serializable;
import java.util.Date;
import java.util.HashMap;

public class Commit implements Serializable {
    private String _parentID;
    private String _commitMessage;
    private Date _commitDate;
    private String _commitSha1;
    private int _length;
    private HashMap<String, String> _commitFiles = null;
    HashMap<File, Integer> myFiles;
    private String _branch = "master";

    public Date getCommitDate(){
        return _commitDate;
    }
    public void setCommitDate(Date d){
        this._commitDate = d;
    }
    public String getCommitMessage(){
        return _commitMessage;
    }
    public String getCommitSha1(){ return _commitSha1;}
    public String getBranch(){ return _branch;}
    public HashMap<String, String> getCommitFiles(){return _commitFiles;}
    public String get_parentID(){return _parentID;}
    public int getLength(){return _length;}

    public Commit(String parentID, String _commitMessage, HashMap<String, String> commitFiles, String branch) {
        this._parentID = parentID;
        this._commitMessage = _commitMessage;
        this._commitDate = new Date();
        this._commitFiles = commitFiles;
        this._branch = branch;
        this._commitSha1 = this.toString();
    }

    public String toString(){ return Utils.sha1(Utils.serialize(this)); }

    public String getParentID() {
        return _parentID;
    }

    /** get the distance between this commit and the init commit.
     * @return the distance.*/
//    public int length() {
//        if (_parentID == null) {
//            return 0;
//        } else {
//            return Main.getCommit(_parentID).length() + 1;
//        }
//    }

    /** get the LENth previous commit of this.
     * @param len the distance.
     * @return that previous commit.*/
    public String shorten(int len) {
        if(len==0){
            return this.getCommitSha1();
        }
        else{
            return Main.getCommit(this.get_parentID()).shorten(len--);
        }
    }
}

