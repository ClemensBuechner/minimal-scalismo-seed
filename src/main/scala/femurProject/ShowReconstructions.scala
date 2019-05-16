package femurProject

import java.awt
import java.awt.Color
import java.io.File

import scalismo.geometry.{EuclideanVector, _3D}
import scalismo.io.MeshIO
import scalismo.registration.TranslationTransform
import scalismo.ui.api.ScalismoUI

import scala.io.StdIn

object ShowReconstructions {

  def main(args: Array[String]): Unit = {

    scalismo.initialize()
    val ui = ScalismoUI()


    val partialFiles = new File("data/femora/partial/").listFiles().sorted
    val reconstructionFiles = new File("data/femora/reconstructions").listFiles().sorted

    (0 until 10).foreach { i: Int =>
      val group = ui.createGroup("reconstruction " + i)
      val translation = TranslationTransform[_3D](EuclideanVector(i * 150, 0, 0))
//      val shiftUp = TranslationTransform[_3D](EuclideanVector(0, 0, 500))

      val partial = MeshIO.readMesh(partialFiles(i)).get
      val partialFit = partial.transform(translation)
//      val partialSeparate = partialFit.transform(shiftUp)
//      val reconstruction = MeshIO.readMesh(reconstructionFiles(i)).get.transform(translation)

      val partialView = ui.show(group, partialFit, "partial")
//      val separateView = ui.show(group, partialSeparate, "partail separate")
//      separateView.color = Color.GREEN
//      val reconstructionView = ui.show(group, reconstruction, "reconstruction")
//      reconstructionView.color = Color.ORANGE

      //      StdIn.readLine("Show next reconstruction")
      //      partialView.opacity = 0
      //      reconstructionView.opacity = 0
    }
  }
}
