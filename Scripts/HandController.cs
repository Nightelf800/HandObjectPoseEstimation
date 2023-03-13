using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using LitJson;

public class HandController : MonoBehaviour
{
    public bool isRight = true;
    public float handScale = 1;
    private JsonData jsonData;
    private List<GameObject> handObjectList = new List<GameObject>();
    private List<HandJoints> handJointsData = new List<HandJoints>();

    private string[] fingerType = { "thumb", "index", "middle", "pinky", "ring" };
    private int[] jointsOrder = { 16, 15, 14, 13, 0, 1, 4, 10, 7, 2, 5, 11, 8, 3, 6, 12, 9, 17, 18, 19, 20 };
    private const int fingerNum = 5;
    private const int jointNum = 4;
    

    // Start is called before the first frame update
    void Start()
    {
        string handType = isRight ? "r" : "l";
        handObjectList.Add(transform.Find("hand_root_r").gameObject);
        for (int i = 0; i < fingerNum; i++)
            for (int j = 1; j <= jointNum; j++)
                handObjectList.Add(transform.Find(fingerType[i] + "_0" + j.ToString() + "_" + handType).gameObject);
        
        /*for (int i = 0; i < 21; i++)
            handObjectList.Add(transform.Find("HandJoint" + i.ToString()).gameObject);*/


        string path = Application.dataPath+"/Resource/eval_ho3dv2_clasbased_artiboost_SUBMIT.json";
        jsonData = JsonMapper.ToObject(File.ReadAllText(path));

        for(int i=0;i<jsonData.Count;i++){
            for(int j=0;j<jsonData[i].Count;j++){
                for(int k=0;k<jsonData[i][j].Count;k++){
                    HandJoints handJoints = new();
                    handJoints.handjoint = new Vector3(float.Parse(jsonData[i][j][k][0].ToString()) * handScale, 
                        float.Parse(jsonData[i][j][k][1].ToString()) * handScale, float.Parse(jsonData[i][j][k][2].ToString()) * handScale);
                    handJointsData.Add(handJoints);
                }
            }
        }

        for(int i = 0; i < handObjectList.Count; i++)
        {
            handObjectList[i].transform.localPosition = handJointsData[jointsOrder[i]].handjoint;
        }

               
        

        
        
    }
}


/*
    <_TargetFrameworkDirectories>non_empty_path_generated_by_unity.rider.package</_TargetFrameworkDirectories>
    <_FullFrameworkReferenceAssemblyPaths>non_empty_path_generated_by_unity.rider.package</_FullFrameworkReferenceAssemblyPaths>
    <DisableHandlePackageFileConflicts>true</DisableHandlePackageFileConflicts>
*/