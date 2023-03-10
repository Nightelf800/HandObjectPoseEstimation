using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using System.IO;
using LitJson;

public class HandController : MonoBehaviour
{
    private GameObject hand;
    private JsonData jsonData;
    private List<List<List<decimal>>> handJointsList;
    // Start is called before the first frame update
    void Start()
    {
        hand = GetComponent<Transform>().gameObject;
        string path = Application.dataPath+"/Resource/eval_ho3dv2_clasbased_artiboost_SUBMIT.json";
        var file = File.ReadAllText(path);


        jsonData = JsonMapper.ToObject(file);

        for(int i=0;i<jsonData.Count;i++){
            for(int j=0;j<jsonData[i].Count;j++){
                for(int k=0;k<jsonData[i][j].Count;k++){
                    Debug.Log(jsonData[i][j][k]);
                }
            }
        }

        
        // for(int i=0;i<jsonData[0][0].Count;i++){
        //     for(int j=0;j<jsonData[0][0][i].Count;j++){
        //         Debug.Log("坐標："+jsonData[0][0][0][i][j]);
        //     }
        // }        
        

        
        
    }
}


/*
    <_TargetFrameworkDirectories>non_empty_path_generated_by_unity.rider.package</_TargetFrameworkDirectories>
    <_FullFrameworkReferenceAssemblyPaths>non_empty_path_generated_by_unity.rider.package</_FullFrameworkReferenceAssemblyPaths>
    <DisableHandlePackageFileConflicts>true</DisableHandlePackageFileConflicts>
*/