package ex1

import scalismo.io.{MeshIO, StatismoIO}
import java.io.File

import scalismo.geometry.Point3D
import scalismo.mesh.TriangleMesh3D
import scalismo.registration.LandmarkRegistration
import scalismo.ui.api.ScalismoUI

object ProcrustesAlignment {

  val sampleSize = 10

  def main(args: Array[String]) {

    implicit val rng = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val faceModel = StatismoIO.readStatismoMeshModel(new File("datasets/bfm.h5")).get

    val samples = (0 until sampleSize).map(_ => faceModel.sample)
    var mean = MeshIO.readMesh(new File("datasets/Paola.stl")).get
    ui.show(mean, "initial mean face")

    var diff = 1d
    var iter = 0
    while (diff > 1E-5) {
      val updateMean: TriangleMesh3D = procrustesStep(samples, mean)
      ui.show(updateMean, "mean " + iter)

      diff = mean.pointSet.points.zip(updateMean.pointSet.points).map(p => (p._1 - p._2).norm).sum
      println("iteration " + iter + ", diff=" + diff)

      mean = updateMean
      iter += 1
    }
    println("procrustes converged")
  }

  def procrustesStep(samples: IndexedSeq[TriangleMesh3D], reference: TriangleMesh3D): TriangleMesh3D = {

    val alignments = samples.map(s => {
      val lmSample = s.pointSet.points
      val lmRef = reference.pointSet.points
      s.transform(LandmarkRegistration.rigid3DLandmarkRegistration(lmSample.zip(lmRef).toSeq, new Point3D(0, 0, 0)))
    })
    val mean = TriangleMesh3D(alignments.map(a => a.pointSet.points.toSeq).transpose.map(a =>
      a.reduce((a, b) => a + b.toVector)).map(a => (a.toVector / sampleSize).toPoint), reference.triangulation)
    mean
  }
}