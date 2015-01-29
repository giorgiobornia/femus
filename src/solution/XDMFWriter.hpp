/*=========================================================================

 Program: FEMUS
 Module: XDMFWriter
 Authors: Eugenio Aulisa, Simone Bnà, Giorgio Bornia

 Copyright (c) FEMTTU
 All rights reserved.

 This software is distributed WITHOUT ANY WARRANTY; without even
 the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
 PURPOSE.  See the above copyright notice for more information.

=========================================================================*/

#ifndef __xdmfwriter_h_
#define __xdmfwriter_h_

//----------------------------------------------------------------------------
// includes :
//----------------------------------------------------------------------------
#include "Writer.hpp"
#include "MultiLevelMeshTwo.hpp"

namespace femus {

class DofMap;
class MultiLevelMeshTwo;
class SystemTwo;



class XDMFWriter : public Writer {

public:

    /** Constructor. */
    XDMFWriter(MultiLevelSolution& ml_sol);
    
    /** Destructor */
    virtual ~XDMFWriter();

    /** write output function */
    virtual void write_system_solutions(const std::string output_path, const char order[], std::vector<std::string>& vars, const unsigned time_step = 0);

    /** write a wrapper file for paraview to open all the files of an history together */
    void write_solution_wrapper(const std::string output_path, const char type[]) const;

  //==================    
   static void write_system_solutions_bc(const std::string namefile, const MultiLevelMeshTwo* mesh, const DofMap* dofmap, const SystemTwo* eqn, const int* bc, int** bc_fe_kk);      
   static void write_system_solutions(const std::string namefile, const MultiLevelMeshTwo* mesh, const DofMap* dofmap, const SystemTwo* eqn);   ///prints on a "Quadratic-Linearized" Mesh //TODO this should be PrintNumericVector of the equation //Writer//
   static void  read_system_solutions(const std::string namefile, const MultiLevelMeshTwo* mesh, const DofMap* dofmap, const SystemTwo* eqn);                       ///read from a "Quadratic-Linearized" Mesh                                      //Writer/Reader// 
    
  //hdf5 ------------------------------------
   static hid_t print_Dhdf5(hid_t file,const std::string & name, hsize_t* dimsf,double* data);
   static hid_t print_Ihdf5(hid_t file,const std::string & name, hsize_t* dimsf,int* data);
   static hid_t print_UIhdf5(hid_t file,const std::string & name, hsize_t* dimsf,uint* data);
   static hid_t read_Dhdf5(hid_t file,const std::string & name,double* data);
   static hid_t read_Ihdf5(hid_t file,const std::string & name,int* data);
   static hid_t read_UIhdf5(hid_t file,const std::string & name,uint* data);

  /** MESH PRINTING */
   static void PrintXDMFAttribute(std::ofstream& outstream, 
				      std::string hdf5_filename, 
				      std::string hdf5_field,
				      std::string attr_name,
				      std::string attr_type,
				      std::string attr_center,
				      std::string data_type,
				      int data_dim_row,
				      int data_dim_col
 				    );
  
  static void PrintXDMFTopology(std::ofstream& outfstream,
				     std::string hdf5_file,
				     std::string hdf5_field,
				     std::string top_type,
				     int top_dim,
				     int datadim_n_elems,
				     int datadim_el_nodes
				    );  
  
  static void PrintXDMFGeometry(std::ofstream& outfstream,
				     std::string hdf5_file,
				     std::string hdf5_field,
				     std::string coord_lev,
				     std::string geom_type,
				     std::string data_type,
				     int data_dim_one,
				     int data_dim_two); 
  
  static void PrintMultimeshXdmfBiquadratic(const std::string output_path, const MultiLevelMeshTwo & mesh);
  
  static void PrintXDMFAllLEVAllVBLinear(const std::string output_path, const MultiLevelMeshTwo & mesh);
  
  static void PrintXDMFGridVBLinear(std::ofstream& out, std::ostringstream& top_file,
			      std::ostringstream& geom_file,
			      const uint Level,
			      const uint vb,
			      const MultiLevelMeshTwo & mesh);
  
  static void PrintXDMFTopologyGeometryLinear(std::ofstream& out,const unsigned Level, const unsigned vb, const MultiLevelMeshTwo& mesh);

  static void PrintSubdomFlagOnCellsBiquadratic(const int vb, const int Level, std::string filename, const MultiLevelMeshTwo & mesh);
  
  static void PrintSubdomFlagOnCellsLinear(std::string filename, const MultiLevelMeshTwo & mesh);
  
  static void PrintAllLEVAllVBLinear(const std::string output_path, const MultiLevelMeshTwo& mesh);
  
  static void PrintConnAllLEVAllVBLinear(const std::string output_path, const MultiLevelMeshTwo& mesh);
  
  static void PrintConnVBLinear(hid_t file, const uint Level, const uint vb, const MultiLevelMeshTwo& mesh); 

  static void PrintElemVBBiquadratic(hid_t file, const uint vb, const std::vector<int> & nd_libm_fm, ElemStoBase** el_sto_in, const std::vector<std::pair<int,int> >  el_fm_libm_in, const MultiLevelMeshTwo & mesh);  
  
  static void ReadMeshFileAndNondimensionalizeBiquadratic(const std::string output_path, MultiLevelMeshTwo & mesh);

  static void PrintMeshFileBiquadratic(const std::string output_path, const MultiLevelMeshTwo & mesh);
  
  /** MATRIX PRINTING */
  static void PrintOneVarMatrixHDF5(const std::string & name, const std::string & groupname, uint** n_nodes_all, int count,int* Mat,int* len,int* len_off,int type1, int type2, int* FELevel );
  
  static void PrintOneVarMGOperatorHDF5(const std::string & filename,const std::string & groupname, uint* n_dofs_lev, int count, int* Op_pos,double* Op_val,int* len,int* len_off, int FELevel_row, int FELevel_col, int fe);
  
// HDF5 FIELDS ===============
   static const std::string _nodes_name; //name for the HDF5 dataset
   static const std::string _elems_name;   //name for the HDF5 dataset
//     std::string _nd_coord_folder;  //TODO why seg fault if I use them?!?
//     std::string _el_pid_name;
//     std::string _nd_map_FineToLev;

private:
  
   static const std::string type_el[4][6];
    

};


} //end namespace femus



#endif
