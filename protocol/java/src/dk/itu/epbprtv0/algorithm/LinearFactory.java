package dk.itu.epbprtv0.algorithm;

import dk.itu.epbprtv0.types.*;
import dk.itu.epbprtv0.distance.*;

import java.util.List;
import java.util.ArrayList;
import java.util.Comparator;

public class LinearFactory implements QueryAlgorithmFactory {
  private static final Comparator<DistanceMetric.DistancePair> _comp =
      new Comparator<DistanceMetric.DistancePair>() {
    @Override
    public int compare(
        DistanceMetric.DistancePair a, DistanceMetric.DistancePair b) {
      if (a.distance < b.distance){
        return -1;
      } else if (a.distance > b.distance) {
        return 1;
      } else return 0;
    }
  };

  private enum PointType {
    UNSPECIFIED,
    BIT,
    DOUBLE_COSINE,
    DOUBLE_EUCLIDEAN
  }
  private class LinearAlgorithm<D> implements PreparedQueryAlgorithm {
    private final TypeInfo<D> t;
    private final DistanceMetric<D> m;
    private ArrayList<D> items = new ArrayList<D>();

    public LinearAlgorithm(TypeInfo<D> t, DistanceMetric<D> m) {
      this.t = t;
      this.m = m;
    }

    @Override
    public boolean train(String item) {
      return items.add(t.parse(item));
    }

    @Override
    public void endTrain() {
    }

    private D preparedQuery = null;
    @Override
    public void prepareQuery(String item_) {
      preparedQuery = t.parse(item_);
    }

    @Override
    public List<Integer> query(String item_, int k) {
      D qp = (preparedQuery == null ? t.parse(item_) : preparedQuery);
      ArrayList<DistanceMetric.DistancePair> distances =
          new ArrayList<DistanceMetric.DistancePair>();
      for (int i = 0; i < items.size(); i++)
        distances.add(new DistanceMetric.DistancePair(i,
            m.getDistance(qp, items.get(i))));
      distances.sort(_comp);
      ArrayList<Integer> results = new ArrayList<Integer>();
      for (DistanceMetric.DistancePair dp :
          distances.subList(0, Math.min(k, distances.size())))
        results.add(dp.index);
      return results;
    }
  }
  private <Q> LinearAlgorithm<Q> make(TypeInfo<Q> t, DistanceMetric<Q> m) {
    return new LinearAlgorithm<Q>(t, m);
  }

  private PointType pt = PointType.UNSPECIFIED;
  
  @Override
  public boolean configure(String var, String val) {
    if (var.equals("point-type")) {
      if (val.equals("bit")) {
        pt = PointType.BIT;
        return true;
      } else if (val.equals("double-cosine")) {
        pt = PointType.DOUBLE_COSINE;
        return true;
      } else if (val.equals("double-euclidean")) {
        pt = PointType.DOUBLE_EUCLIDEAN;
        return true;
      } else return false;
    } else return false;
  }

  @Override
  public QueryAlgorithm endConfigure() {
    switch (pt) {
    case BIT:
      return make(BitInfo.INSTANCE, HammingDistance.INSTANCE);
    case DOUBLE_COSINE:
      return make(DoubleInfo.INSTANCE, CosineDistance.INSTANCE);
    case DOUBLE_EUCLIDEAN:
      return make(DoubleInfo.INSTANCE, EuclideanDistance.INSTANCE);
    default:
      return null;
    }
  }
}
