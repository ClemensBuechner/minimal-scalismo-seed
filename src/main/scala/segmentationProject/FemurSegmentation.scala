package segmentationProject

import java.io.File

import scalismo.io.{ImageIO, MeshIO, StatisticalModelIO}
import scalismo.ui.api.ScalismoUI

object FemurSegmentation {

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)
    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val model = StatisticalModelIO.readStatisticalMeshModel(new java.io.File(dataDir +
      "femur-asm.h5"))

    val tests = Array(4, 14, 23, 25, 30)
    val targets = Array(1, 9, 10, 13, 37)
    val testCTs = tests.map { i: Int =>
      ImageIO.read3DScalarImage[Short](new File(dataDir + "test/" + i + ".nii"))
    }
    val testReferences = tests.map { i: Int =>
      MeshIO.readMesh(new File(dataDir + "test/" + i + ".stl"))
    }
    val targetCTs = targets.map { i: Int =>
      ImageIO.read3DScalarImage[Short](new File(dataDir + "targets/" + i + ".nii"))
    }

    
  }
}
