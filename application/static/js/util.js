my_assert = function (flag, msg) {
  if (!flag) {
    throw msg;
  }
};

my_round = (x) => x.toFixed(5);

my_round_percent = (x) => (Math.round(x * 100) + '%');


function hsl2rgb(h, s, l) {
  let r;
  let g;
  let b;

  if (s == 0) {
    r = g = b = l; // achromatic
  } else {
    const hue2rgb = function hue2rgb(p, q, t) {
      if (t < 0) t += 1;
      if (t > 1) t -= 1;
      if (t < 1 / 6) return p + (q - p) * 6 * t;
      if (t < 1 / 2) return q;
      if (t < 2 / 3) return p + (q - p) * (2 / 3 - t) * 6;
      return p;
    };

    const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
    const p = 2 * l - q;
    r = hue2rgb(p, q, h + 1 / 3);
    g = hue2rgb(p, q, h);
    b = hue2rgb(p, q, h - 1 / 3);
  }

  return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
}

function rgb2hsl(r, g, b) {
  r /= 255, g /= 255, b /= 255;
  const max = Math.max(r, g, b);
  const min = Math.min(r, g, b);
  let h;
  let s;
  const l = (max + min) / 2;

  if (max == min) {
    h = s = 0; // achromatic
  } else {
    const d = max - min;
    s = l > 0.5 ? d / (2 - max - min) : d / (max + min);
    switch (max) {
      case r:
        h = (g - b) / d + (g < b ? 6 : 0);
        break;
      case g:
        h = (b - r) / d + 2;
        break;
      case b:
        h = (r - g) / d + 4;
        break;
    }
    h /= 6;
  }

  return [h, s, l];
}

function rgb2hex(col) {
  col = col.slice(1);
  const num = parseInt(col, 16);
  const r = (num >> 16);
  const g = ((num >> 8) & 0xFF);
  const b = (num & 0xFF);
  return [r, g, b];
}

function hex2color(r, g, b) {
  return '#' + (b | (g << 8) | (r << 16)).toString(16);
}

LightenDarkenColor = function (col) {
  const rgb = rgb2hex(col);
  const hsl = rgb2hsl(rgb[0], rgb[1], rgb[2]);
  const rgb1 = hsl2rgb(hsl[0], hsl[1], 0.55);
  const rgb2 = hsl2rgb(hsl[0], hsl[1], 0.75);
  return [hex2color(rgb1[0], rgb1[1], rgb1[2]), hex2color(rgb2[0], rgb2[1], rgb2[2])];
};

SVGUtil = {};
SVGUtil.ensureSpecificSVGItem = function (vn, type, cls) {
  let item = undefined;
  if (cls === undefined) {
    item = vn.select(type);
    if (item.empty()) {
      item = vn.append(type);
    }
  } else {
    item = vn.select(type + '.' + cls);
    if (item.empty()) {
      item = vn.append(type).attr('class', cls);
    }
  }
  return item;
};
SVGUtil.attrD3 = function (vnode, info) {
  for (const x in info) {
    vnode = vnode.attr(x, info[x]);
  }
  return vnode;
};
SVGUtil.styleD3 = function (vnode, info) {
  for (const x in info) {
    vnode = vnode.style(x, info[x]);
  }
  return vnode;
};

SVGUtil.drawClosedShape = function (node_list) {
  let s = '';
  node_list.forEach((node, i) => {
    s += ((i == 0) ? 'M' : 'L');
    s += ('' + node.x + ' ' + node.y + ' ');
  });
  return (s + 'Z');
};

SVGUtil.CustomGlyph = {};
SVGUtil.CustomGlyph.circle = function (vn, center, size, animate) {
  let circle = SVGUtil.ensureSpecificSVGItem(vn, 'circle');
  if (animate) {
    circle = animate(circle);
  }
  circle = SVGUtil.attrD3(
    circle, {
      cx: center.x,
      cy: center.y,
      r: size,
    });
  circle = SVGUtil.styleD3(circle, {
    'fill': 'transparent',
    'stroke': 'black',
    'stroke-width': 1,
  });
  return circle;
};

SVGUtil.CustomGlyph.tri = function (vn, center, size, animate) {
  let tri = SVGUtil.ensureSpecificSVGItem(vn, 'path');
  if (animate) {
    tri = animate(tri);
  }
  tri.attr('d', d3.line()([
    [center.x, center.y - size],
    [center.x + size * 0.866, center.y + size / 2],
    [center.x - size * 0.866, center.y + size / 2],
  ]) + 'Z');
  tri = SVGUtil.styleD3(tri, {
    'fill': 'transparent',
    'stroke': 'black',
    'stroke-width': 1,
  });
  return tri;
};

SVGUtil.CustomGlyph.cross = function (vn, center, size, animate) {
  let cross = SVGUtil.ensureSpecificSVGItem(vn, 'path');
  if (animate) {
    cross = animate(cross);
  }
  cross.attr('d', `${d3.line()([
    [center.x, center.y - size],
    [center.x, center.y + size],
  ])}${d3.line()([
    [center.x - size, center.y],
    [center.x + size, center.y],
  ])}`)
  cross = SVGUtil.styleD3(cross, {
    'fill': 'transparent',
    'stroke': 'black',
    'stroke-width': 1,
  });
  return cross;
};

SVGUtil.CustomGlyph.criss = function (vn, center, size, animate) {
  let criss = SVGUtil.ensureSpecificSVGItem(vn, 'path');
  if (animate) {
    criss = animate(criss);
  }
  criss.attr('d', `${d3.line()([
    [center.x - size, center.y - size],
    [center.x + size, center.y + size],
  ])}${d3.line()([
    [center.x - size, center.y + size],
    [center.x + size, center.y - size],
  ])}`)
  criss = SVGUtil.styleD3(criss, {
    'fill': 'transparent',
    'stroke': 'black',
    'stroke-width': 1,
  });
  return criss;
};

SVGUtil.CustomGlyph.rect = function (vn, center, size, animate) {
  let rect = SVGUtil.ensureSpecificSVGItem(vn, 'rect');
  if (animate) {
    rect = animate(rect);
  }
  rect = SVGUtil.attrD3(
    rect, {
      x: center.x - size / 2,
      y: center.y - size / 2,
      width: size,
      height: size,
    });
  rect = SVGUtil.styleD3(rect, {
    'fill': 'transparent',
    'stroke': 'black',
    'stroke-width': 1,
  });
  return rect;
};


SVGUtil.getFormatAssigner = function (format_list) {
  const that = {};
  that.formats = format_list;
  that.key2format = {};

  that.get = function (key) {
    if (key.includes('_')) key = key.split('_')[0];
    //if (key[key.length-1] === '2') key = key.substr(0, key.length-1)
    if (that.key2format[key] === undefined) {
      const used = {};
      for (const k in that.key2format) {
        used[that.key2format[k]] = true;
      }
      let fms = d3.range(that.formats.length).filter((ii) => (used[ii] === undefined));
      if (fms.length == 0) {
        return undefined;
      } else {
        fms = fms[0];
        that.key2format[key] = fms;
      }
    }
    return that.formats[that.key2format[key]];
  };

  that.remove = function (key) {
    if (key.includes('_')) key = key.split('_')[0];
    //if (key[key.length - 1] === '2') key = key.substr(0, key.length - 1)
    if (that.key2format[key] !== undefined) {
      delete that.key2format[key];
    }
  };

  return that;
};

function getTextWidth(text, font) {
  let canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
  let context = canvas.getContext("2d");
  context.font = font;
  return context.measureText(text).width;
}