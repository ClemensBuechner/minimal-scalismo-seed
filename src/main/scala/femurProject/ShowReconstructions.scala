package femurProject

import java.awt.Color
import java.io.File

import scalismo.io.MeshIO
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
      val partial = MeshIO.readMesh(partialFiles(i)).get
      val reconstruction = MeshIO.readMesh(reconstructionFiles(i)).get

      val partialView = ui.show(group, partial, "partial")
      partialView.color = Color.GREEN
      val reconstructionView = ui.show(group, reconstruction, "reconstruction")

      StdIn.readLine("Show next reconstruction")
      partialView.opacity = 0
      reconstructionView.opacity = 0
    }
  }
}
