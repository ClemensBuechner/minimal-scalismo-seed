package segmentationProject

import java.io.File

import scalismo.io.ActiveShapeModelIO
import scalismo.statisticalmodel.StatisticalMeshModel
import scalismo.ui.api.ScalismoUI

import scala.io.StdIn

object LandmarksGenerator {

  def getLandmarksForModel(ui: ScalismoUI, model: StatisticalMeshModel): Unit = {
    val modelGroup = ui.createGroup("model")
    val modelView = ui.show(modelGroup, model, "model")

    StdIn.readLine("Click landmarks in the model, save them, and press [Enter].")
  }

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val dataDir = "data/handedData/"

    val asm = ActiveShapeModelIO.readActiveShapeModel(new File(dataDir + "femur-asm.h5")).get
    getLandmarksForModel(ui, asm.statisticalModel)
  }
}
