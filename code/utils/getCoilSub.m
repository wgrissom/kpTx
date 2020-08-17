
            function coilSub = getCoilSub(nc)
            % For indexing coil combinations in AhA.
            [ich, ic] = ndgrid(1:nc);
            ltMask = ich>=ic; % lower triangle mask
            coilSub = [ich(ltMask), ic(ltMask)];
            end
