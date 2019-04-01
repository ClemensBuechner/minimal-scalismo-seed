package ex2

import java.io.File

import scalismo.common.{DiscreteField, NearestNeighborInterpolator, PointId, UnstructuredPointsDomain}
import scalismo.geometry.{EuclideanVector, _3D}
import scalismo.io.MeshIO
import scalismo.io.StatismoIO.StatismoModelType
import scalismo.statisticalmodel.{DiscreteLowRankGaussianProcess, StatisticalMeshModel}
import scalismo.ui.api.ScalismoUI

object GaussianProcess {

  def main(args: Array[String]): Unit = {

    implicit val rng = scalismo.utils.Random(42)

    scalismo.initialize()
    val ui = ScalismoUI()

    val files = new File("data/femora/aligned/").listFiles()
    val dataset = files.map{f => MeshIO.readMesh(f).get}

    val reference = dataset.head
    val defFields = dataset.tail.map { m =>
      val deformationVectors = m.pointSet.pointIds.map { id : PointId =>
        val p = m.pointSet.point(id)
        p - reference.pointSet.findClosestPoint(p).point
      }.toIndexedSeq

      DiscreteField[_3D, UnstructuredPointsDomain[_3D], EuclideanVector[_3D]](m.pointSet, deformationVectors)
    }

    val interpolator = NearestNeighborInterpolator[_3D, EuclideanVector[_3D]]
    val contiuousField = defFields.map( f => f.interpolate(interpolator))
    val gp = DiscreteLowRankGaussianProcess.createUsingPCA(reference.pointSet, contiuousField)
    val model = StatisticalMeshModel(reference, gp.interpolate(interpolator))
    ui.show(model, "mean")
  }
}
