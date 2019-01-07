function [vertex, faces] = read_obj(filename)
% from BoffinBlogger

fid = fopen(filename);
if fid<0
    error(['Cannot open ' filename '.']);
end
[str, count] = fread(fid, [1,inf], 'uint8=>char'); 
%fprintf('Read %d characters from %s\n', count, filename);
fclose(fid);

vertex_lines = regexp(str,'v [^\n]*\n', 'match'); % v need to be at begin of a line
vertex = zeros(length(vertex_lines), 3);
i = 1;
for k = 1: length(vertex_lines)
    v = sscanf(vertex_lines{k}, 'v %f %f %f');
    if (length(v) == 3)
        vertex(i, :) = v';
        i = i + 1;
        continue;
    end
    % if reached here: regexp must be wrong..
    vertex(i,:) = [];
end

face_lines = regexp(str,'f [^\n]*\n', 'match'); % f need to be at begin of a line
faces = zeros(length(face_lines), 3);
i = 1;
for k = 1:length(face_lines)
    f = sscanf(face_lines{k}, 'f %d//%d %d//%d %d//%d');
    if (length(f) == 6) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(3);
        faces(i, 3) = f(5);
        i = i + 1;
        continue
    end
    f = sscanf(face_lines{k}, 'f %d %d %d');
    if (length(f) == 3) % face
        faces(i, :) = f';
        i = i + 1;
        continue
    end
    f = sscanf(face_lines{k}, 'f %d/%d %d/%d %d/%d');
    if (length(f) == 6) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(3);
        faces(i, 3) = f(5);
        i = i + 1;
        continue
    end
    f = sscanf(face_lines{k}, 'f %d/%d/%d %d/%d/%d %d/%d/%d');
    if (length(f) == 9) % face
        faces(i, 1) = f(1);
        faces(i, 2) = f(4);
        faces(i, 3) = f(7);
        i = i + 1;
        continue
    end
    % if reached here: regexp must be wrong..
    faces(i,:) = [];
end
