function lightcontrol(label)
pos1 = [2 4 1 1];
pos2 = [4 4 1 1];
pos3 = [6 4 1 1];
pos4 = [8 4 1 1];
pos5 = [10 4 1 1];
pos6 = [12 4 1 1];

switch label
    case 1
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','green')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
    case 2
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','green')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
    case 3
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','green')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
    case 4
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','green')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
    case 5
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','green')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','white')
    case 6
        rectangle('Position',pos1,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos2,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos3,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos4,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos5,'Curvature',[1 1],'FaceColor','white')
        rectangle('Position',pos6,'Curvature',[1 1],'FaceColor','green')
end
end